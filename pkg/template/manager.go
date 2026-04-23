package template

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	defaultOwner = "john221wick"
	defaultRepo  = "genxcode-go"
	defaultBranch = "main"
	templatesPath = "templates"
)

// Manager handles fetching and caching templates.
type Manager struct {
	CacheDir    string
	RemoteBase  string
	HTTPClient  *http.Client
}

// NewManager creates a template manager with default settings.
func NewManager() *Manager {
	cacheDir := defaultCacheDir()
	return &Manager{
		CacheDir:   cacheDir,
		RemoteBase: fmt.Sprintf("https://raw.githubusercontent.com/%s/%s/%s/%s", defaultOwner, defaultRepo, defaultBranch, templatesPath),
		HTTPClient: &http.Client{Timeout: 30 * time.Second},
	}
}

// NewManagerWithRepo allows customizing the remote repo.
func NewManagerWithRepo(owner, repo, branch string) *Manager {
	cacheDir := defaultCacheDir()
	return &Manager{
		CacheDir:   cacheDir,
		RemoteBase: fmt.Sprintf("https://raw.githubusercontent.com/%s/%s/%s/%s", owner, repo, branch, templatesPath),
		HTTPClient: &http.Client{Timeout: 30 * time.Second},
	}
}

func defaultCacheDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, ".genxcode", "templates")
}

// TemplateDir returns the local path for a template.
func (m *Manager) TemplateDir(name string) string {
	return filepath.Join(m.CacheDir, name)
}

// IsCached checks if a template exists locally.
func (m *Manager) IsCached(name string) bool {
	_, err := os.Stat(m.TemplateDir(name))
	return err == nil
}

// Fetch downloads a template from the remote repo.
func (m *Manager) Fetch(name string) error {
	dir := m.TemplateDir(name)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("mkdir: %w", err)
	}

	// Download manifest
	manifestURL := fmt.Sprintf("%s/%s/%s", m.RemoteBase, name, "template.yaml")
	manifestPath := filepath.Join(dir, "template.yaml")
	if err := m.downloadFile(manifestURL, manifestPath); err != nil {
		return fmt.Errorf("fetch manifest: %w", err)
	}

	// Read manifest to discover files directory
	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		return fmt.Errorf("read manifest: %w", err)
	}

	filesDir := "files"
	for _, line := range strings.Split(string(manifestData), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "files_dir:") {
			filesDir = strings.TrimSpace(strings.TrimPrefix(line, "files_dir:"))
			filesDir = strings.Trim(filesDir, `"'`)
		}
	}

	// Discover and download all files recursively
	filesBasePath := filepath.Join(dir, filesDir)
	return m.fetchDirectory(fmt.Sprintf("%s/%s/%s", m.RemoteBase, name, filesDir), filesBasePath)
}

// fetchDirectory downloads all files in a remote directory using GitHub API.
func (m *Manager) fetchDirectory(remoteURL, localPath string) error {
	// Convert raw URL to API URL for listing
	apiURL := rawToAPI(remoteURL)
	
	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("X-GitHub-Api-Version", "2022-11-28")

	resp, err := m.HTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("list directory: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 404 {
		// Directory might not exist, try as single file
		return m.downloadFile(remoteURL, localPath)
	}
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("list directory %s: %s - %s", apiURL, resp.Status, string(body))
	}

	var items []struct {
		Type string `json:"type"`
		Name string `json:"name"`
		Path string `json:"path"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&items); err != nil {
		return fmt.Errorf("decode listing: %w", err)
	}

	for _, item := range items {
		itemLocalPath := filepath.Join(localPath, item.Name)
		itemRemoteURL := remoteURL + "/" + item.Name
		if item.Type == "dir" {
			if err := m.fetchDirectory(itemRemoteURL, itemLocalPath); err != nil {
				return err
			}
		} else {
			if err := os.MkdirAll(filepath.Dir(itemLocalPath), 0755); err != nil {
				return err
			}
			if err := m.downloadFile(itemRemoteURL, itemLocalPath); err != nil {
				return err
			}
		}
	}
	return nil
}

func rawToAPI(rawURL string) string {
	// https://raw.githubusercontent.com/owner/repo/branch/path
	// -> https://api.github.com/repos/owner/repo/contents/path
	parts := strings.SplitN(rawURL, "/", 6)
	if len(parts) < 6 {
		return rawURL
	}
	// parts[0] = https:
	// parts[1] = ""
	// parts[2] = raw.githubusercontent.com
	// parts[3] = owner
	// parts[4] = repo
	// parts[5] = branch/path...
	branchAndPath := parts[5]
	branchPathParts := strings.SplitN(branchAndPath, "/", 2)
	branch := branchPathParts[0]
	path := ""
	if len(branchPathParts) > 1 {
		path = branchPathParts[1]
	}
	return fmt.Sprintf("https://api.github.com/repos/%s/%s/contents/%s?ref=%s", parts[3], parts[4], path, branch)
}

func (m *Manager) downloadFile(url, dest string) error {
	resp, err := m.HTTPClient.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("HTTP %s: %s", resp.Status, string(body))
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0755); err != nil {
		return err
	}
	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, resp.Body)
	return err
}

// ListCached returns names of locally cached templates.
func (m *Manager) ListCached() ([]string, error) {
	entries, err := os.ReadDir(m.CacheDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() {
			names = append(names, e.Name())
		}
	}
	return names, nil
}

// ListAvailable fetches all template names from the remote repo.
func (m *Manager) ListAvailable() ([]string, error) {
	apiURL := rawToAPI(m.RemoteBase)

	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("X-GitHub-Api-Version", "2022-11-28")

	resp, err := m.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("list remote templates: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		// Fallback to cached list if remote fails
		return m.ListCached()
	}

	var items []struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&items); err != nil {
		return m.ListCached()
	}

	var names []string
	for _, item := range items {
		if item.Type == "dir" {
			names = append(names, item.Name)
		}
	}

	// Also ensure they're cached so we can read specs
	for _, name := range names {
		if !m.IsCached(name) {
			// Fetch just the manifest for description
			dir := m.TemplateDir(name)
			os.MkdirAll(dir, 0755)
			manifestURL := fmt.Sprintf("%s/%s/%s", m.RemoteBase, name, "template.yaml")
			manifestPath := filepath.Join(dir, "template.yaml")
			m.downloadFile(manifestURL, manifestPath)
		}
	}

	return names, nil
}

// ClearCache removes all cached templates.
func (m *Manager) ClearCache() error {
	return os.RemoveAll(m.CacheDir)
}

// EnsureAvailable checks cache and fetches if missing.
func (m *Manager) EnsureAvailable(name string) error {
	if m.IsCached(name) {
		return nil
	}
	fmt.Printf("Template %s not cached. Downloading...\n", name)
	return m.Fetch(name)
}
