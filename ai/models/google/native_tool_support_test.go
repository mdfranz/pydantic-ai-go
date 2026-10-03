package google_test

import (
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/google"
)

func TestModelRejectsUnpreparedSpeech(t *testing.T) {
	model := google.NewModel("gemini-3-pro")
	for _, history := range [][]ai.ModelMessage{
		{ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{Speaker: ai.SpeechSpeakerUser}}}},
		{ai.ModelResponse{Parts: []ai.ResponsePart{ai.SpeechPart{Speaker: ai.SpeechSpeakerAssistant}}}},
	} {
		if _, err := model.Request(t.Context(), history, ai.ModelRequestParams{}); !errors.Is(
			err, ai.ErrUnpreparedSpeech,
		) {
			t.Fatalf("Google accepted unprepared speech: %v", err)
		}
	}
}

func TestNativeToolSupport(t *testing.T) {
	model := google.NewModel("gemini-3-pro")
	if profile := model.ModelProfile(); profile.SupportsImageOutput || profile.DefaultOutputMode != ai.OutputModeTool {
		t.Fatalf("unexpected text model profile: %#v", profile)
	}
	if !model.SupportsNativeTool(ai.WebSearchTool{}) || !model.SupportsNativeTool(ai.FileSearchTool{
		FileStoreIDs: []string{"store"},
	}) || model.SupportsNativeTool(ai.ImageGenerationTool{}) || model.SupportsNativeTool(ai.MemoryTool{}) ||
		model.SupportsNativeTool((*ai.WebSearchTool)(nil)) {
		t.Fatal("unexpected Google native-tool support")
	}
	image := google.NewModel("gemini-3-pro-image-preview")
	if profile := image.ModelProfile(); !profile.SupportsImageOutput || profile.DefaultOutputMode != ai.OutputModeTool {
		t.Fatalf("unexpected image model profile: %#v", profile)
	}
	if !image.SupportsNativeTool(ai.ImageGenerationTool{}) {
		t.Fatal("image model did not report image-generation support")
	}
}

func TestWebSearchSupportMatchesRenderedOptions(t *testing.T) {
	falseValue := false
	model := google.NewModel("gemini-3-pro")
	unsupported := []ai.WebSearchTool{
		{AllowedDomains: []string{"go.dev"}},
		{BlockedDomains: []string{"example.com"}},
		{MaxUses: 1},
		{ExternalWebAccess: &falseValue},
		{UserLocation: &ai.WebSearchUserLocation{Country: "NL"}},
		{SearchContextSize: ai.WebSearchContextHigh},
	}
	for _, tool := range unsupported {
		if model.SupportsNativeTool(tool) || model.SupportsNativeTool(&tool) {
			t.Fatalf("Google reported unsupported web search configuration as supported: %#v", tool)
		}
	}
	if !model.SupportsNativeTool(ai.WebSearchTool{}) || !model.SupportsNativeTool(
		&ai.WebSearchTool{SearchContextSize: ai.WebSearchContextMedium},
	) {
		t.Fatal("Google rejected its supported web search configuration")
	}
}

func TestWebSearchUnsupportedOptionsFailOrUseFallback(t *testing.T) {
	requests := 0
	var body map[string]any
	model := newNamedServer(t, "gemini-3-pro", func(response http.ResponseWriter, request *http.Request) {
		requests++
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`))
	})

	for _, tool := range []ai.NativeTool{
		ai.WebSearchTool{AllowedDomains: []string{"go.dev"}},
		&ai.WebSearchTool{MaxUses: 1},
	} {
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{tool}}); err == nil {
			t.Fatalf("unsupported web search reached the provider: %#v", tool)
		}
	}
	if requests != 0 {
		t.Fatalf("unsupported web search made %d provider requests", requests)
	}

	for _, optional := range []ai.NativeTool{
		ai.WebSearchTool{AllowedDomains: []string{"go.dev"}, Optional: true},
		&ai.WebSearchTool{AllowedDomains: []string{"go.dev"}, Optional: true},
	} {
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{optional}}); err != nil {
			t.Fatalf("optional unsupported web search did not omit itself: %v", err)
		}
		if body["tools"] != nil {
			t.Fatalf("optional unsupported web search reached the provider: %#v", body)
		}
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{&ai.WebSearchTool{}},
	}); err != nil {
		t.Fatalf("supported pointer web search failed: %v", err)
	}
	if body["tools"] == nil {
		t.Fatalf("supported pointer web search was omitted: %#v", body)
	}

	web := ai.WebSearchTool{AllowedDomains: []string{"go.dev"}}
	if _, err := ai.RequestModel(t.Context(), model, nil, ai.ModelRequestParams{
		Tools:       []ai.ToolDefinition{{Name: "local_search", Schema: map[string]any{"type": "object"}, NativeFallbackFor: web.UniqueID()}},
		NativeTools: []ai.NativeTool{web},
	}); err != nil {
		t.Fatalf("unsupported web search did not use its local fallback: %v", err)
	}
	tools, ok := body["tools"].([]any)
	if !ok || len(tools) != 1 || tools[0].(map[string]any)["functionDeclarations"] == nil {
		t.Fatalf("local fallback was not sent to Google: %#v", body)
	}
}
