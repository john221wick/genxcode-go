package cmd

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/john221wick/genxcode-go/pkg/config"
	"github.com/john221wick/genxcode-go/pkg/generator"
	"github.com/john221wick/genxcode-go/pkg/template"
	"github.com/manifoldco/promptui"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

const logo = `
   ██████╗ ███████╗███╗   ██╗██╗  ██╗ ██████╗ ██████╗ ██████╗ ███████╗
  ██╔════╝ ██╔════╝████╗  ██║╚██╗██╔╝██╔════╝██╔═══██╗██╔══██╗██╔════╝
  ██║  ███╗█████╗  ██╔██╗ ██║ ╚███╔╝ ██║     ██║   ██║██║  ██║█████╗
  ██║   ██║██╔══╝  ██║╚██╗██║ ██╔██╗ ██║     ██║   ██║██║  ██║██╔══╝
  ╚██████╔╝███████╗██║ ╚████║██╔╝ ██╗╚██████╗╚██████╔╝██████╔╝███████╗
   ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
                    Boilerplate Code Generator
`

func printLogo() {
	fmt.Print(logo)
}

var (
	forceFlag bool
	repoOwner string
	repoName  string
	repoBranch string
)

func newManager() *template.Manager {
	if repoOwner != "" && repoName != "" {
		branch := repoBranch
		if branch == "" {
			branch = "main"
		}
		return template.NewManagerWithRepo(repoOwner, repoName, branch)
	}
	return template.NewManager()
}

var rootCmd = &cobra.Command{
	Use:     "genxcode",
	Short:   "Generate boilerplate code from remote templates",
	Long:    `genxcode fetches templates from GitHub, caches them locally, and renders projects from YAML configs.`,
	Version: config.Version,
}

var initCmd = &cobra.Command{
	Use:   "init <template>",
	Short: "Initialize a project from a template",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		printLogo()
		templateName := strings.TrimSpace(args[0])
		return runWizard(templateName)
	},
}

// setupMode represents how the user wants to set up the project.
type setupMode struct {
	Name        string
	Description string
}

func runWizard(templateName string) error {
	mgr := newManager()

	if err := mgr.EnsureAvailable(templateName); err != nil {
		return fmt.Errorf("failed to fetch template: %w", err)
	}

	tmplDir := mgr.TemplateDir(templateName)
	spec, err := config.LoadTemplateSpec(filepath.Join(tmplDir, "template.yaml"))
	if err != nil {
		return err
	}

	// Load default config as ordered key-value pairs
	defaultConfigPath := filepath.Join(tmplDir, spec.DefaultConfigFilename)
	keys, defaults, err := loadOrderedYAML(defaultConfigPath)
	if err != nil {
		return fmt.Errorf("load defaults: %w", err)
	}

	// Step 1: Ask project name
	namePrompt := promptui.Prompt{
		Label:   "Project name",
		Default: defaults["name"],
	}
	projectName, err := namePrompt.Run()
	if err != nil {
		return handlePromptErr(err)
	}
	projectName = strings.TrimSpace(projectName)
	if projectName == "" {
		projectName = templateName
	}
	defaults["name"] = projectName
	fmt.Println()

	// Step 2: Choose setup mode
	modes := []setupMode{
		{Name: "Generate with defaults", Description: "Use default settings, generate project now"},
		{Name: "Configure interactively", Description: "Customize each option, then generate"},
		{Name: "Save config file only", Description: "Create genxcode.yaml for manual editing"},
	}

	modeTemplates := &promptui.SelectTemplates{
		Active:   "  > {{ .Name | cyan | bold }}  {{ .Description | faint }}",
		Inactive: "    {{ .Name | white }}  {{ .Description | faint }}",
		Selected: "  * {{ .Name | green | bold }}",
		Help:     " ",
	}

	modePrompt := promptui.Select{
		Label:     "How would you like to set up?",
		Items:     modes,
		Templates: modeTemplates,
		Size:      3,
		HideHelp:  true,
	}

	modeIdx, _, err := modePrompt.Run()
	if err != nil {
		return handlePromptErr(err)
	}
	fmt.Println()

	switch modeIdx {
	case 0: // Generate with defaults
		parsed := make(map[string]interface{})
		for k, v := range defaults {
			parsed[k] = parseValue(v)
		}
		parsed["name"] = projectName
		return generateProject(mgr, templateName, spec, parsed)

	case 1: // Configure interactively
		configData, err := interactiveConfig(keys, defaults, templateName)
		if err != nil {
			return err
		}
		return generateProject(mgr, templateName, spec, configData)

	case 2: // Save config only
		configData, err := interactiveConfig(keys, defaults, templateName)
		if err != nil {
			return err
		}
		return saveConfigFile(templateName, configData, spec)
	}
	return nil
}

// interactiveConfig prompts user for each field, showing defaults.
func interactiveConfig(keys []string, defaults map[string]string, templateName string) (map[string]interface{}, error) {
	scanner := bufio.NewScanner(os.Stdin)
	result := make(map[string]interface{})

	fmt.Println("  Configure your project (press Enter to keep default):")
	fmt.Println()

	for _, key := range keys {
		// Skip template and name — already set
		if key == "template" || key == "name" {
			result[key] = defaults[key]
			continue
		}

		defVal := defaults[key]
		fmt.Printf("  \033[36m%s\033[0m \033[2m(%s)\033[0m: ", key, defVal)

		scanner.Scan()
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			input = defVal
		}

		result[key] = parseValue(input)
	}
	fmt.Println()
	return result, nil
}

// parseValue tries to convert string input to appropriate Go type.
func parseValue(s string) interface{} {
	// Bool
	if s == "true" {
		return true
	}
	if s == "false" {
		return false
	}

	// Comma-separated list
	if strings.Contains(s, ",") {
		parts := strings.Split(s, ",")
		// Try as int list
		allInt := true
		var ints []interface{}
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if n, err := strconv.Atoi(p); err == nil {
				ints = append(ints, n)
			} else {
				allInt = false
				break
			}
		}
		if allInt && len(ints) > 0 {
			return ints
		}
		// String list
		var strs []interface{}
		for _, p := range parts {
			strs = append(strs, strings.TrimSpace(p))
		}
		return strs
	}

	// Int
	if n, err := strconv.Atoi(s); err == nil {
		return n
	}

	// Float
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return f
	}

	return s
}

// generateProject builds the project from config data directly.
func generateProject(mgr *template.Manager, templateName string, spec *config.TemplateSpec, configData map[string]interface{}) error {
	// Merge with defaults
	tmplDir := mgr.TemplateDir(templateName)
	defaultPath := filepath.Join(tmplDir, spec.DefaultConfigFilename)
	defaultData, err := loadYAMLMap(defaultPath)
	if err != nil {
		return fmt.Errorf("load defaults: %w", err)
	}
	merged := mergeMaps(defaultData, configData)
	merged["template"] = templateName

	name := "project"
	if n, ok := merged["name"].(string); ok && n != "" {
		name = n
	}
	merged["name"] = name

	ctx, err := generator.BuildContext(merged)
	if err != nil {
		return fmt.Errorf("build context: %w", err)
	}

	outputDir := filepath.Join(".", name)
	fmt.Printf("  Generating %s (%s) in %s\n\n", name, templateName, outputDir)
	if err := generator.GenerateProject(tmplDir, outputDir, ctx); err != nil {
		return fmt.Errorf("generate: %w", err)
	}

	fmt.Printf("\n  Done! Project created at ./%s\n", name)
	if spec.ApplyHint != "" {
		fmt.Printf("  %s\n", spec.ApplyHint)
	}
	return nil
}

// saveConfigFile writes genxcode.yaml from collected config data.
func saveConfigFile(templateName string, configData map[string]interface{}, spec *config.TemplateSpec) error {
	configData["template"] = templateName
	dest := config.DefaultConfigFilename

	if _, err := os.Stat(dest); err == nil && !forceFlag {
		return fmt.Errorf("%s already exists. Use --force to overwrite", dest)
	}

	out, err := yaml.Marshal(configData)
	if err != nil {
		return fmt.Errorf("marshal config: %w", err)
	}

	content := fmt.Sprintf("# Generated by `genxcode init %s`\n%s", templateName, string(out))
	if err := os.WriteFile(dest, []byte(content), 0644); err != nil {
		return fmt.Errorf("write config: %w", err)
	}

	fmt.Printf("  Created %s\n", dest)
	fmt.Println("  Run `genxcode apply` to generate your project.")
	if spec.ApplyHint != "" {
		fmt.Printf("  %s\n", spec.ApplyHint)
	}
	return nil
}

// loadOrderedYAML reads a YAML file and returns keys in order + values as strings.
func loadOrderedYAML(path string) ([]string, map[string]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}

	// Parse with yaml.v3 Node to preserve key order
	var node yaml.Node
	if err := yaml.Unmarshal(data, &node); err != nil {
		return nil, nil, err
	}

	if node.Kind != yaml.DocumentNode || len(node.Content) == 0 {
		return nil, nil, fmt.Errorf("invalid YAML structure")
	}

	mapping := node.Content[0]
	if mapping.Kind != yaml.MappingNode {
		return nil, nil, fmt.Errorf("expected mapping at top level")
	}

	var keys []string
	vals := make(map[string]string)

	for i := 0; i+1 < len(mapping.Content); i += 2 {
		keyNode := mapping.Content[i]
		valNode := mapping.Content[i+1]
		key := keyNode.Value
		keys = append(keys, key)

		switch valNode.Kind {
		case yaml.SequenceNode:
			// Format as comma-separated
			var items []string
			for _, item := range valNode.Content {
				items = append(items, item.Value)
			}
			vals[key] = strings.Join(items, ", ")
		default:
			vals[key] = valNode.Value
		}
	}

	return keys, vals, nil
}

func handlePromptErr(err error) error {
	if err == promptui.ErrInterrupt || err == promptui.ErrEOF {
		fmt.Println("\nCancelled.")
		os.Exit(0)
	}
	return err
}

var applyCmd = &cobra.Command{
	Use:   "apply",
	Short: "Generate boilerplate from genxcode.yaml",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg, err := config.LoadConfig(config.DefaultConfigFilename)
		if err != nil {
			return err
		}

		mgr := newManager()
		if err := mgr.EnsureAvailable(cfg.Template); err != nil {
			return fmt.Errorf("failed to fetch template: %w", err)
		}

		tmplDir := mgr.TemplateDir(cfg.Template)
		spec, err := config.LoadTemplateSpec(filepath.Join(tmplDir, "template.yaml"))
		if err != nil {
			return err
		}

		// Merge defaults
		defaultPath := filepath.Join(tmplDir, spec.DefaultConfigFilename)
		defaultData, err := loadYAMLMap(defaultPath)
		if err != nil {
			return fmt.Errorf("load defaults: %w", err)
		}
		merged := mergeMaps(defaultData, cfg.Data)
		merged["template"] = cfg.Template
		merged["name"] = cfg.Name

		ctx, err := generator.BuildContext(merged)
		if err != nil {
			return fmt.Errorf("build context: %w", err)
		}

		outputDir := filepath.Join(".", cfg.Name)
		fmt.Printf("Generating %s (%s) in %s\n", cfg.Name, cfg.Template, outputDir)
		if err := generator.GenerateProject(tmplDir, outputDir, ctx); err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		fmt.Println()
		if spec.ApplyHint != "" {
			fmt.Printf("  %s\n", spec.ApplyHint)
		}
		return nil
	},
}

var updateCmd = &cobra.Command{
	Use:   "update",
	Short: "Update genxcode to the latest release",
	RunE: func(cmd *cobra.Command, args []string) error {
		return selfUpdate()
	},
}

func selfUpdate() error {
	currentVer := strings.TrimPrefix(config.Version, "v")
	if currentVer == "" || currentVer == "dev" {
		fmt.Println("You are running a development build. Skipping self-update.")
		return nil
	}

	// Fetch latest release tag
	apiURL := "https://api.github.com/repos/john221wick/genxcode-go/releases/latest"
	resp, err := http.Get(apiURL)
	if err != nil {
		return fmt.Errorf("failed to check for updates: %w", err)
	}
	defer resp.Body.Close()

	var release struct {
		TagName string `json:"tag_name"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&release); err != nil {
		return fmt.Errorf("failed to parse release info: %w", err)
	}

	latestVer := strings.TrimPrefix(release.TagName, "v")
	if latestVer == currentVer {
		fmt.Printf("genxcode is already up to date (%s).\n", currentVer)
		return nil
	}

	fmt.Printf("Current: v%s  Latest: v%s\n", currentVer, latestVer)

	// Detect OS/arch
	goos := runtime.GOOS
	goarch := runtime.GOARCH
	if goarch == "amd64" {
		goarch = "amd64"
	}

	ext := "tar.gz"
	if goos == "windows" {
		ext = "zip"
	}

	assetName := fmt.Sprintf("genxcode_%s_%s_%s.%s", latestVer, goos, goarch, ext)
	downloadURL := fmt.Sprintf("https://github.com/john221wick/genxcode-go/releases/download/v%s/%s", latestVer, assetName)

	fmt.Printf("Downloading %s...\n", assetName)

	// Download to temp file
	tmpFile, err := os.CreateTemp("", "genxcode-update-*")
	if err != nil {
		return fmt.Errorf("create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())

	dlResp, err := http.Get(downloadURL)
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	defer dlResp.Body.Close()
	if dlResp.StatusCode != 200 {
		return fmt.Errorf("download failed: HTTP %s (asset may not exist for your platform)", dlResp.Status)
	}

	if _, err := io.Copy(tmpFile, dlResp.Body); err != nil {
		return fmt.Errorf("write download: %w", err)
	}
	tmpFile.Close()

	// Extract if archive
	extractedPath := tmpFile.Name()
	if ext == "tar.gz" {
		extractDir, err := os.MkdirTemp("", "genxcode-extract-*")
		if err != nil {
			return fmt.Errorf("create extract dir: %w", err)
		}
		defer os.RemoveAll(extractDir)

		cmd := exec.Command("tar", "-xzf", tmpFile.Name(), "-C", extractDir)
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("extract archive: %w", err)
		}
		extractedPath = filepath.Join(extractDir, "genxcode")
	}

	if err := os.Chmod(extractedPath, 0755); err != nil {
		return fmt.Errorf("chmod: %w", err)
	}

	// Find current binary path
	currentPath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("find current binary: %w", err)
	}
	currentPath, err = filepath.EvalSymlinks(currentPath)
	if err != nil {
		return fmt.Errorf("resolve binary path: %w", err)
	}

	// Replace binary: rename old, move new
	oldPath := currentPath + ".old"
	if err := os.Rename(currentPath, oldPath); err != nil {
		return fmt.Errorf("backup old binary: %w", err)
	}
	if err := os.Rename(extractedPath, currentPath); err != nil {
		// Try to restore old
		os.Rename(oldPath, currentPath)
		return fmt.Errorf("install new binary: %w", err)
	}
	os.Remove(oldPath)

	fmt.Printf("Updated genxcode to v%s\n", latestVer)
	return nil
}

type templateItem struct {
	Name        string
	Description string
	Aliases     string
}

var listCmd = &cobra.Command{
	Use:   "list",
	Short: "Browse and select from available templates",
	RunE: func(cmd *cobra.Command, args []string) error {
		printLogo()
		mgr := newManager()

		// Fetch available template names from remote
		available, err := mgr.ListAvailable()
		if err != nil {
			return fmt.Errorf("failed to list templates: %w", err)
		}
		if len(available) == 0 {
			fmt.Println("No templates available.")
			return nil
		}

		// Build items with descriptions (fetch specs for cached templates)
		items := make([]templateItem, 0, len(available))
		for _, name := range available {
			item := templateItem{Name: name}
			tmplDir := mgr.TemplateDir(name)
			spec, err := config.LoadTemplateSpec(filepath.Join(tmplDir, "template.yaml"))
			if err == nil {
				if spec.Description != "" {
					item.Description = truncate(spec.Description, 50)
				}
				if len(spec.Aliases) > 0 {
					item.Aliases = strings.Join(spec.Aliases, ", ")
				}
			}
			items = append(items, item)
		}

		// Interactive selection
		templates := &promptui.SelectTemplates{
			Label:    "{{ . }}",
			Active:   "  > {{ .Name | cyan | bold }}{{ if .Description }}  {{ .Description | faint }}{{ end }}",
			Inactive: "    {{ .Name | white }}{{ if .Description }}  {{ .Description | faint }}{{ end }}",
			Selected: "  * {{ .Name | green | bold }}",
			Help:     " ",
		}

		prompt := promptui.Select{
			Label:             "Select a template (↑/↓ to move, enter to select)",
			Items:             items,
			Templates:         templates,
			Size:              10,
			HideHelp:          true,
			StartInSearchMode: false,
		}

		idx, _, err := prompt.Run()
		if err != nil {
			if err == promptui.ErrInterrupt || err == promptui.ErrEOF {
				fmt.Println("Selection cancelled.")
				return nil
			}
			return fmt.Errorf("prompt failed: %w", err)
		}

		selected := items[idx]
		fmt.Println()
		return runWizard(selected.Name)
	},
}

var removeCmd = &cobra.Command{
	Use:   "remove",
	Short: "Remove genxcode.yaml from current directory",
	RunE: func(cmd *cobra.Command, args []string) error {
		path := config.DefaultConfigFilename
		if _, err := os.Stat(path); os.IsNotExist(err) {
			return fmt.Errorf("%s does not exist", path)
		}
		if err := os.Remove(path); err != nil {
			return fmt.Errorf("failed to remove %s: %w", path, err)
		}
		fmt.Printf("Removed %s\n", path)
		return nil
	},
}

func init() {
	rootCmd.PersistentFlags().StringVar(&repoOwner, "owner", "", "GitHub owner for remote templates")
	rootCmd.PersistentFlags().StringVar(&repoName, "repo", "", "GitHub repo for remote templates")
	rootCmd.PersistentFlags().StringVar(&repoBranch, "branch", "", "GitHub branch for remote templates")

	initCmd.Flags().BoolVar(&forceFlag, "force", false, "Overwrite existing config")

	rootCmd.AddCommand(initCmd)
	rootCmd.AddCommand(applyCmd)
	rootCmd.AddCommand(updateCmd)
	rootCmd.AddCommand(listCmd)
	rootCmd.AddCommand(removeCmd)
}

// Execute runs the root command.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func loadYAMLMap(path string) (map[string]interface{}, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m map[string]interface{}
	if err := config.ParseYAML(data, &m); err != nil {
		return nil, err
	}
	return m, nil
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

func mergeMaps(base, overlay map[string]interface{}) map[string]interface{} {
	out := make(map[string]interface{})
	for k, v := range base {
		out[k] = v
	}
	for k, v := range overlay {
		out[k] = v
	}
	return out
}
