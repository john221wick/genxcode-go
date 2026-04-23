package generator

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"text/template"
)

// BuildContext takes raw config data and returns an enriched map with
// computed fields added. Works for any template — all fields from the
// YAML config are accessible directly (e.g. {{.KernelName}}, {{.Name}}).
func BuildContext(data map[string]interface{}) (map[string]interface{}, error) {
	ctx := make(map[string]interface{})
	for k, v := range data {
		ctx[k] = v
	}

	// Ensure basic fields
	if _, ok := ctx["Name"]; !ok {
		ctx["Name"] = getString(data, "name", "project")
	}
	if _, ok := ctx["Template"]; !ok {
		ctx["Template"] = getString(data, "template", "")
	}

	// Capitalize common fields so templates can use either .name or .Name
	ctx["Name"] = getString(data, "name", "project")
	ctx["Template"] = getString(data, "template", "")

	// Compute pytorch-specific fields if relevant
	arch := getString(data, "arch", "")
	if arch != "" {
		ctx["Arch"] = arch
	}

	inputDim := getInt(data, "input_dim", 0)
	outputDim := getInt(data, "output_dim", 0)
	hidden := getIntSlice(data, "hidden", nil)
	activation := getString(data, "activation", "relu")
	metrics := getStringSlice(data, "metrics", nil)

	if inputDim > 0 {
		ctx["InputDim"] = inputDim
	}
	if outputDim > 0 {
		ctx["OutputDim"] = outputDim
	}
	if hidden != nil {
		ctx["Hidden"] = hidden
	}
	if activation != "" {
		ctx["Activation"] = activation
	}
	if metrics != nil {
		ctx["Metrics"] = metrics
	}

	ctx["Prod"] = getBool(data, "prod", false)
	if lr := getFloat(data, "lr", 0); lr > 0 {
		ctx["LR"] = lr
	}
	if epochs := getInt(data, "epochs", 0); epochs > 0 {
		ctx["Epochs"] = epochs
	}
	if bs := getInt(data, "batch_size", 0); bs > 0 {
		ctx["BatchSize"] = bs
	}
	if device := getString(data, "device", ""); device != "" {
		ctx["Device"] = device
	}

	// Capitalize all snake_case keys to PascalCase for template access
	// e.g. kernel_name -> KernelName, vector_size -> VectorSize
	for k, v := range data {
		pascal := snakeToPascal(k)
		if pascal != k {
			ctx[pascal] = v
		}
	}

	// Compute metric booleans
	for _, m := range metrics {
		switch m {
		case "f1":
			ctx["HasF1"] = true
		case "precision":
			ctx["HasPrecision"] = true
		case "recall":
			ctx["HasRecall"] = true
		case "accuracy":
			ctx["HasAccuracy"] = true
		case "loss":
			ctx["HasLoss"] = true
		}
	}

	needsSklearn := getBool(ctx, "HasF1", false) || getBool(ctx, "HasPrecision", false) || getBool(ctx, "HasRecall", false)
	ctx["NeedsSklearn"] = needsSklearn

	var sklearnFuncs []string
	if getBool(ctx, "HasF1", false) {
		sklearnFuncs = append(sklearnFuncs, "f1_score")
	}
	if getBool(ctx, "HasPrecision", false) {
		sklearnFuncs = append(sklearnFuncs, "precision_score")
	}
	if getBool(ctx, "HasRecall", false) {
		sklearnFuncs = append(sklearnFuncs, "recall_score")
	}
	ctx["SklearnImports"] = strings.Join(sklearnFuncs, ", ")

	// Build layers for MLP models
	if inputDim > 0 && hidden != nil {
		type Layer struct {
			In  int
			Out int
		}
		prev := inputDim
		var layers []Layer
		for _, h := range hidden {
			layers = append(layers, Layer{In: prev, Out: h})
			prev = h
		}
		ctx["Layers"] = layers
		ctx["OutputLayerIn"] = prev
	}

	return ctx, nil
}

// snakeToPascal converts snake_case to PascalCase.
func snakeToPascal(s string) string {
	parts := strings.Split(s, "_")
	for i, p := range parts {
		if len(p) > 0 {
			parts[i] = strings.ToUpper(p[:1]) + p[1:]
		}
	}
	return strings.Join(parts, "")
}

// TemplateFuncs returns custom functions available in templates.
func TemplateFuncs() template.FuncMap {
	return template.FuncMap{
		"json": func(v interface{}) string {
			b, _ := json.Marshal(v)
			return string(b)
		},
		"contains": contains,
		"join": func(sep string, vals []string) string {
			return strings.Join(vals, sep)
		},
		"last": func(i, len int) bool {
			return i == len-1
		},
		"upper": strings.ToUpper,
		"split": func(s string, sep string) []string {
			return strings.Split(s, sep)
		},
	}
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// Render renders a single template string with the context map.
func Render(tmplStr string, ctx map[string]interface{}) (string, error) {
	tmpl, err := template.New("fragment").Funcs(TemplateFuncs()).Parse(tmplStr)
	if err != nil {
		return "", fmt.Errorf("parse template: %w", err)
	}
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, ctx); err != nil {
		return "", fmt.Errorf("execute template: %w", err)
	}
	return buf.String(), nil
}

// GenerateProject walks template files and renders them to output directory.
func GenerateProject(templateDir, outputDir string, ctx map[string]interface{}) error {
	filesDir := filepath.Join(templateDir, "files")
	return filepath.Walk(filesDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}

		rel, err := filepath.Rel(filesDir, path)
		if err != nil {
			return err
		}

		isTemplate := strings.HasSuffix(rel, ".j2") || strings.HasSuffix(rel, ".tmpl")
		targetRel := rel
		if isTemplate {
			targetRel = targetRel[:len(targetRel)-3]
		}

		renderedRel, err := Render(targetRel, ctx)
		if err != nil {
			return fmt.Errorf("render filename: %w", err)
		}

		targetPath := filepath.Join(outputDir, renderedRel)
		if err := os.MkdirAll(filepath.Dir(targetPath), 0755); err != nil {
			return err
		}

		if isTemplate {
			content, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			rendered, err := Render(string(content), ctx)
			if err != nil {
				return fmt.Errorf("render %s: %w", rel, err)
			}
			if err := os.WriteFile(targetPath, []byte(rendered), 0644); err != nil {
				return err
			}
		} else {
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			if err := os.WriteFile(targetPath, data, 0644); err != nil {
				return err
			}
		}

		fmt.Printf("  %s\n", renderedRel)
		return nil
	})
}

// Helper getters

func getString(m map[string]interface{}, key, def string) string {
	v, ok := m[key]
	if !ok {
		return def
	}
	s, ok := v.(string)
	if ok {
		return s
	}
	return fmt.Sprintf("%v", v)
}

func getInt(m map[string]interface{}, key string, def int) int {
	v, ok := m[key]
	if !ok {
		return def
	}
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	}
	return def
}

func getFloat(m map[string]interface{}, key string, def float64) float64 {
	v, ok := m[key]
	if !ok {
		return def
	}
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	case int64:
		return float64(n)
	}
	return def
}

func getBool(m map[string]interface{}, key string, def bool) bool {
	v, ok := m[key]
	if !ok {
		return def
	}
	b, ok := v.(bool)
	if ok {
		return b
	}
	return def
}

func getIntSlice(m map[string]interface{}, key string, def []int) []int {
	v, ok := m[key]
	if !ok {
		return def
	}
	slice, ok := v.([]interface{})
	if !ok {
		return def
	}
	var out []int
	for _, item := range slice {
		switch n := item.(type) {
		case int:
			out = append(out, n)
		case int64:
			out = append(out, int(n))
		case float64:
			out = append(out, int(n))
		}
	}
	return out
}

func getStringSlice(m map[string]interface{}, key string, def []string) []string {
	v, ok := m[key]
	if !ok {
		return def
	}
	slice, ok := v.([]interface{})
	if !ok {
		return def
	}
	var out []string
	for _, item := range slice {
		s, ok := item.(string)
		if ok {
			out = append(out, s)
		}
	}
	return out
}
