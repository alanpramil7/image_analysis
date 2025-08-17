package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/table"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/log"
	_ "github.com/mattn/go-sqlite3"
	"github.com/openai/openai-go/v2"
	"google.golang.org/genai"
)

// Global logger
var logger = log.NewWithOptions(os.Stderr, log.Options{
	Level: log.DebugLevel,
})

// Models configuration
var openAIModels = []string{
	"gpt-4o",
	// "gpt-4o-mini",
	"o3-mini",
}

var geminiModels = []string{
	"gemini-2.5-flash",
	"gemini-2.5-pro",
	"gemini-2.5-flash-lite",
	"gemini-2.0-flash",
	"gemini-2.0-flash-lite",
}

// Data structures
type Checklist struct {
	RoomType  string `json:"room_type"`
	Checklist string `json:"checklist"`
}

type ModelResponse struct {
	Answer string `json:"answer"`
	Reason string `json:"reason"`
}

type ImageData struct {
	ImageId  int    `json:"image_id"`
	RoomType string `json:"room_type"`
	ImageUrl string `json:"image_url"`
	Status   bool   `json:"status"`
}

type ModelAccuracy struct {
	Provider        string
	ModelName       string
	RoomType        string
	TotalTests      int
	CorrectAnswers  int
	AccuracyPercent float64
	AvgResponseTime float64
	AvgInputTokens  float64
	AvgOutputTokens float64
	TruePositives   int
	TrueNegatives   int
	FalsePositives  int
	FalseNegatives  int
}

type TestResult struct {
	Provider       string
	ModelName      string
	RoomType       string
	Answer         string
	ExpectedAnswer bool
	Reason         string
	ResponseTime   int64
	InputTokens    int64
	OutputTokens   int64
	TotalTokens    int64
}

// TUI model
type model struct {
	detailedTable table.Model
}

func (m model) Init() tea.Cmd {
	return nil
}

func (m model) View() string {
	var s strings.Builder

	// Header with instructions
	headerStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("205")).
		MarginBottom(1)

	s.WriteString(headerStyle.Render("Room Inspector Benchmark Results"))
	s.WriteString("\n")

	// Table headers
	tableHeaderStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("39"))

	s.WriteString(fmt.Sprintf("%s \n", tableHeaderStyle.Render("Detailed Model Accuracy by Room Type")))
	s.WriteString("\n")
	s.WriteString(m.detailedTable.View())

	return s.String()
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			return m, tea.Quit
		}
	}

	var cmd tea.Cmd
	m.detailedTable, cmd = m.detailedTable.Update(msg)

	return m, cmd
}

// Utility functions
func loadFromJson[T any](filename string) ([]T, error) {
	content, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("error loading json file %s: %v", filename, err)
	}

	var result []T
	if err := json.Unmarshal(content, &result); err != nil {
		return nil, fmt.Errorf("error unmarshalling %s: %v", filename, err)
	}

	return result, nil
}

func downloadImage(imageUrl string) ([]byte, string, error) {
	resp, err := http.Get(imageUrl)
	if err != nil {
		return nil, "", fmt.Errorf("error downloading image: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("error downloading image: status code %d", resp.StatusCode)
	}

	imageBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", fmt.Errorf("error reading image bytes: %v", err)
	}

	mimeType := resp.Header.Get("Content-Type")
	if mimeType == "" {
		return nil, "", fmt.Errorf("error getting image mime type")
	}

	return imageBytes, mimeType, nil
}

// Database functions
func initDB() (*sql.DB, error) {
	db, err := sql.Open("sqlite3", "./data/benchmark.db")
	if err != nil {
		return nil, fmt.Errorf("error opening database: %v", err)
	}

	createTableQuery := `
		CREATE TABLE IF NOT EXISTS results (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			provider TEXT NOT NULL,
			model_name TEXT NOT NULL,
			room_type TEXT NOT NULL,
			checklist TEXT NOT NULL,
			image_url TEXT NOT NULL,
			answer TEXT NOT NULL,
			expected_answer BOOLEAN NOT NULL,
			reason TEXT NOT NULL,
			response_time INTEGER NOT NULL,
			input_tokens INTEGER NOT NULL,
			output_tokens INTEGER NOT NULL,
			total_tokens INTEGER NOT NULL,
			timestamp INTEGER NOT NULL
		)`

	if _, err := db.Exec(createTableQuery); err != nil {
		return nil, fmt.Errorf("error creating database table: %v", err)
	}

	return db, nil
}

func insertTestResult(db *sql.DB, result TestResult, checklist Checklist, imageUrl string) error {
	query := `
		INSERT INTO results (
			provider, model_name, room_type, checklist, image_url, answer, 
			expected_answer, reason, response_time, input_tokens, 
			output_tokens, total_tokens, timestamp
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	_, err := db.Exec(query,
		result.Provider, result.ModelName, result.RoomType, checklist.Checklist,
		imageUrl, result.Answer, result.ExpectedAnswer, result.Reason,
		result.ResponseTime, result.InputTokens, result.OutputTokens,
		result.TotalTokens, time.Now().Unix())

	if err != nil {
		return fmt.Errorf("error inserting test result: %v", err)
	}

	return nil
}

// Model calling functions
func callGeminiModel(db *sql.DB, modelName string, c Checklist, imageData ImageData) error {
	exists, err := testExists(db, "gemini", modelName, c.RoomType, imageData.ImageUrl)
	if err != nil {
		return err
	}
	if exists {
		logger.Infof("Skipping Gemini model %s for %s - already exists in cache", modelName, c.RoomType)
		return nil
	}

	startTime := time.Now()
	logger.Infof("Calling Gemini Model: %s for image id %d", modelName, imageData.ImageId)

	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return fmt.Errorf("error creating Gemini client: %v", err)
	}

	imageBytes, mimeType, err := downloadImage(imageData.ImageUrl)
	if err != nil {
		return fmt.Errorf("error downloading image: %v", err)
	}

	systemMessage := `You are a precise room inspector. You will be provided with a specific room type and checklist to verify after the room has been cleaned. Analyze the image carefully and determine if the room meets the cleaning standards based on the provided checklist. Be concise and definitive in your answer.`

	parts := []*genai.Part{
		genai.NewPartFromBytes(imageBytes, mimeType),
		genai.NewPartFromText(fmt.Sprintf("Room Type: %s\n\nChecklist Requirements:\n%s\n\nBased on the image and checklist above, does this room meet the cleaning standards? Respond with 'true' if it meets standards, 'false' if it doesn't.", c.RoomType, c.Checklist)),
	}

	contents := []*genai.Content{
		genai.NewContentFromParts(parts, genai.RoleUser),
	}

	schema := genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"answer": {Type: genai.TypeString, Enum: []string{"true", "false"}},
			"reason": {Type: genai.TypeString},
		},
		Required: []string{"answer", "reason"},
	}

	config := &genai.GenerateContentConfig{
		SystemInstruction: &genai.Content{Parts: []*genai.Part{
			{Text: systemMessage},
		}},
		ResponseMIMEType: "application/json",
		ResponseSchema:   &schema,
	}

	response, err := client.Models.GenerateContent(ctx, modelName, contents, config)
	if err != nil {
		return fmt.Errorf("error calling Gemini model: %v", err)
	}

	endTime := time.Now()
	responseTime := endTime.Sub(startTime).Milliseconds()

	logger.Debugf("Gemini %s Response: %s", modelName, response.Text())
	logger.Debugf("Tokens Spent for %s: Prompt=%d, Candidates=%d, Total=%d",
		modelName,
		response.UsageMetadata.PromptTokenCount,
		response.UsageMetadata.CandidatesTokenCount,
		response.UsageMetadata.TotalTokenCount)

	var modelResponse ModelResponse
	if err := json.Unmarshal([]byte(response.Text()), &modelResponse); err != nil {
		return fmt.Errorf("error unmarshalling Gemini response: %v", err)
	}

	result := TestResult{
		Provider:       "gemini",
		ModelName:      modelName,
		RoomType:       c.RoomType,
		Answer:         modelResponse.Answer,
		ExpectedAnswer: imageData.Status,
		Reason:         modelResponse.Reason,
		ResponseTime:   responseTime,
		InputTokens:    int64(response.UsageMetadata.PromptTokenCount),
		OutputTokens:   int64(response.UsageMetadata.CandidatesTokenCount),
		TotalTokens:    int64(response.UsageMetadata.TotalTokenCount),
	}

	return insertTestResult(db, result, c, imageData.ImageUrl)
}

func callOpenAIModel(db *sql.DB, modelName string, c Checklist, imageData ImageData) error {
	exists, err := testExists(db, "openai", modelName, c.RoomType, imageData.ImageUrl)
	if err != nil {
		return err
	}

	if exists {
		logger.Infof("Skipping OpenAI model %s for %d - already exists in cache", modelName, imageData.ImageId)
		return nil
	}

	startTime := time.Now()
	logger.Infof("Calling OpenAI Model: %s for image id %d", modelName, imageData.ImageId)

	systemMessage := `You are a precise room inspector. You will be provided with a specific room type and checklist to verify after the room has been cleaned. Analyze the image carefully and determine if the room meets the cleaning standards based on the provided checklist. Be concise and definitive in your answer.`

	client := openai.NewClient()
	ctx := context.Background()

	userMessage := fmt.Sprintf("Room Type: %s\n\nChecklist Requirements:\n%s\n\nBased on the image and checklist above, does this room meet the cleaning standards? Respond with 'true' if it meets standards, 'false' if it doesn't.", c.RoomType, c.Checklist)

	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(systemMessage),
		openai.UserMessage(
			[]openai.ChatCompletionContentPartUnionParam{
				openai.TextContentPart(userMessage),
				openai.ImageContentPart(
					openai.ChatCompletionContentPartImageImageURLParam{URL: imageData.ImageUrl},
				),
			},
		),
	}

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{
				"type": "string",
				"enum": []string{"true", "false"},
			},
			"reason": map[string]any{
				"type": "string",
			},
		},
		"required":             []string{"answer", "reason"},
		"additionalProperties": false,
	}

	schemaParam := openai.ResponseFormatJSONSchemaJSONSchemaParam{
		Name:        c.RoomType,
		Description: openai.String(c.RoomType),
		Schema:      schema,
		Strict:      openai.Bool(true),
	}

	chatCompletionParams := openai.ChatCompletionNewParams{
		Model:    modelName,
		Messages: messages,
		ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{JSONSchema: schemaParam},
		},
	}

	chatCompletion, err := client.Chat.Completions.New(ctx, chatCompletionParams)
	if err != nil {
		return fmt.Errorf("error calling OpenAI model: %v", err)
	}

	endTime := time.Now()
	responseTime := endTime.Sub(startTime).Milliseconds()

	logger.Debugf("OpenAI %s Response: %s", modelName, chatCompletion.Choices[0].Message.Content)
	logger.Debugf("Tokens Spent for %s: Prompt=%d, Completion=%d, Total=%d",
		modelName,
		chatCompletion.Usage.PromptTokens,
		chatCompletion.Usage.CompletionTokens,
		chatCompletion.Usage.TotalTokens)

	var modelResponse ModelResponse
	if err := json.Unmarshal([]byte(chatCompletion.Choices[0].Message.Content), &modelResponse); err != nil {
		return fmt.Errorf("error unmarshalling OpenAI response: %v", err)
	}

	result := TestResult{
		Provider:       "openai",
		ModelName:      modelName,
		RoomType:       c.RoomType,
		Answer:         modelResponse.Answer,
		ExpectedAnswer: imageData.Status,
		Reason:         modelResponse.Reason,
		ResponseTime:   responseTime,
		InputTokens:    int64(chatCompletion.Usage.PromptTokens),
		OutputTokens:   int64(chatCompletion.Usage.CompletionTokens),
		TotalTokens:    int64(chatCompletion.Usage.TotalTokens),
	}

	return insertTestResult(db, result, c, imageData.ImageUrl)
}

// Display functions
func displayDetailedModelAccuracy(db *sql.DB) (table.Model, error) {
	query := `
		SELECT 
			provider,
			model_name,
			room_type,
			COUNT(*) as total_tests,
			SUM(CASE 
				WHEN (answer = 'true' AND expected_answer = 1) OR (answer = 'false' AND expected_answer = 0) 
				THEN 1 ELSE 0 
			END) as correct_answers,
			SUM(CASE 
				WHEN answer = 'true' AND expected_answer = 1 
				THEN 1 ELSE 0 
			END) as true_positives,
			SUM(CASE 
				WHEN answer = 'false' AND expected_answer = 0 
				THEN 1 ELSE 0 
			END) as true_negatives,
			SUM(CASE 
				WHEN answer = 'true' AND expected_answer = 0 
				THEN 1 ELSE 0 
			END) as false_positives,
			SUM(CASE 
				WHEN answer = 'false' AND expected_answer = 1 
				THEN 1 ELSE 0 
			END) as false_negatives,
			AVG(response_time) as avg_response_time,
			AVG(input_tokens) as avg_input_tokens,
			AVG(output_tokens) as avg_output_tokens
		FROM results 
		GROUP BY provider, model_name, room_type
		ORDER BY provider, model_name, room_type`

	rows, err := db.Query(query)
	if err != nil {
		return table.Model{}, fmt.Errorf("error querying detailed accuracy: %v", err)
	}
	defer rows.Close()

	columns := []table.Column{
		{Title: "Provider", Width: 10},
		{Title: "Model", Width: 20},
		{Title: "Room Type", Width: 15},
		{Title: "Total Tests", Width: 10},
		{Title: "Accuracy %", Width: 10},
		{Title: "Correct", Width: 8},
		{Title: "True Pos", Width: 8},
		{Title: "True Neg", Width: 8},
		{Title: "False Pos", Width: 9},
		{Title: "False Neg", Width: 9},
		{Title: "Avg Time(s)", Width: 12},
		{Title: "Avg Tokens", Width: 10},
	}

	var tableRows []table.Row
	for rows.Next() {
		var acc ModelAccuracy
		err := rows.Scan(
			&acc.Provider, &acc.ModelName, &acc.RoomType, &acc.TotalTests,
			&acc.CorrectAnswers, &acc.TruePositives, &acc.TrueNegatives,
			&acc.FalsePositives, &acc.FalseNegatives, &acc.AvgResponseTime,
			&acc.AvgInputTokens, &acc.AvgOutputTokens,
		)
		if err != nil {
			return table.Model{}, fmt.Errorf("error scanning detailed row: %v", err)
		}

		if acc.TotalTests > 0 {
			acc.AccuracyPercent = float64(acc.CorrectAnswers) / float64(acc.TotalTests) * 100
		}

		row := table.Row{
			acc.Provider,
			acc.ModelName,
			acc.RoomType,
			strconv.Itoa(acc.TotalTests),
			fmt.Sprintf("%.2f%%", acc.AccuracyPercent),
			strconv.Itoa(acc.CorrectAnswers),
			strconv.Itoa(acc.TruePositives),
			strconv.Itoa(acc.TrueNegatives),
			strconv.Itoa(acc.FalsePositives),
			strconv.Itoa(acc.FalseNegatives),
			fmt.Sprintf("%.0f", acc.AvgResponseTime),
			fmt.Sprintf("%.0f", acc.AvgInputTokens+acc.AvgOutputTokens),
		}
		tableRows = append(tableRows, row)
	}

	return createStyledTable(columns, tableRows, 15), nil
}

func createStyledTable(columns []table.Column, rows []table.Row, height int) table.Model {
	t := table.New(
		table.WithColumns(columns),
		table.WithRows(rows),
		table.WithFocused(true),
		table.WithHeight(height),
	)

	s := table.DefaultStyles()
	s.Header = s.Header.
		BorderStyle(lipgloss.NormalBorder()).
		BorderForeground(lipgloss.Color("240")).
		BorderBottom(true).
		Bold(true)
	s.Selected = s.Selected.
		Foreground(lipgloss.Color("229")).
		Background(lipgloss.Color("57")).
		Bold(false)
	t.SetStyles(s)

	return t
}

func testExists(db *sql.DB, provider, modelName, roomType, imageUrl string) (bool, error) {
	query := `
		SELECT COUNT(*) FROM results 
		WHERE provider = ? AND model_name = ? AND room_type = ? AND image_url = ?`

	var count int
	err := db.QueryRow(query, provider, modelName, roomType, imageUrl).Scan(&count)
	if err != nil {
		return false, fmt.Errorf("error checking if test exists: %v", err)
	}

	return count > 0, nil
}

// Test execution functions
func runAllTests(db *sql.DB, checklists []Checklist, imageData []ImageData) error {
	logger.Info("Starting benchmark tests...")

	// Test OpenAI models
	for _, modelName := range openAIModels {
		for _, checklist := range checklists {
			for _, imageData := range imageData {
				if err := callOpenAIModel(db, modelName, checklist, imageData); err != nil {
					logger.Errorf("Error testing OpenAI model %s: %v", modelName, err)
					continue // Continue with next test instead of failing completely
				}
			}
		}
	}

	// Test Gemini models
	for _, modelName := range geminiModels {
		for _, checklist := range checklists {
			for _, imageData := range imageData {
				if err := callGeminiModel(db, modelName, checklist, imageData); err != nil {
					logger.Errorf("Error testing Gemini model %s: %v", modelName, err)
					continue // Continue with next test instead of failing completely
				}
			}
		}
	}

	logger.Info("Benchmark tests completed!")
	return nil
}

func main() {
	logger.Info("Starting Room Inspector Benchmark Application")

	// Load checklist data
	checklists, err := loadFromJson[Checklist]("./data/checklist.json")
	if err != nil {
		logger.Fatal("Failed to load checklist data", "error", err)
	}
	logger.Infof("Loaded %d checklists", len(checklists))

	// Load Image data
	imageData, err := loadFromJson[ImageData]("./data/data.json")
	if err != nil {
		logger.Fatal("Failed to load image data", "error", err)
	}

	// Initialize database
	db, err := initDB()
	if err != nil {
		logger.Fatal("Failed to initialize database", "error", err)
	}
	defer db.Close()
	logger.Info("Database initialized successfully")

	// Run all tests
	if err := runAllTests(db, checklists, imageData); err != nil {
		logger.Fatal("Failed to run tests", "error", err)
	}

	detailedTable, err := displayDetailedModelAccuracy(db)
	if err != nil {
		logger.Fatal("Failed to generate detailed results table", "error", err)
	}

	// Create model with both tables
	m := model{
		detailedTable: detailedTable,
	}

	// Start TUI
	program := tea.NewProgram(m, tea.WithAltScreen())
	if _, err := program.Run(); err != nil {
		logger.Fatal("Failed to start TUI", "error", err)
	}

	logger.Info("Application completed successfully")
}
