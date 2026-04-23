#!/bin/sh
set -e

REPO="john221wick/genxcode-go"
BINARY="genxcode"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$ARCH" in
    x86_64) ARCH="amd64" ;;
    arm64|aarch64) ARCH="arm64" ;;
    *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

case "$OS" in
    linux|darwin) ;;
    *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

# Fetch latest release tag
API_URL="https://api.github.com/repos/$REPO/releases/latest"
API_RESPONSE=$(curl -s "$API_URL")
TAG=$(echo "$API_RESPONSE" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
if [ -z "$TAG" ]; then
    echo "Error: Failed to fetch latest release tag."
    echo ""
    echo "GitHub API response:"
    echo "$API_RESPONSE"
    echo ""
    echo "This usually means:"
    echo "  1. The repo $REPO does not exist on GitHub yet"
    echo "  2. The repo exists but has no releases"
    echo ""
    echo "To fix: create the repo, push code, and run 'make publish'"
    exit 1
fi

echo "Installing $BINARY $TAG for ${OS}_${ARCH}..."

# Download binary
FILENAME="${BINARY}_${TAG}_${OS}_${ARCH}.tar.gz"
URL="https://github.com/$REPO/releases/download/$TAG/$FILENAME"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "Downloading $FILENAME..."
HTTP_CODE=$(curl -sSL -w "%{http_code}" "$URL" -o "$TMPDIR/${BINARY}.tar.gz")

if [ "$HTTP_CODE" != "200" ]; then
    echo ""
    echo "Error: Download failed with HTTP $HTTP_CODE"
    echo "URL: $URL"
    echo ""
    echo "This usually means the release asset does not exist."
    echo "Available assets for $TAG:"
    curl -s "$API_URL" | grep '"name":' | sed -E 's/.*"name": "([^"]+)".*/  - \1/' || true
    echo ""
    echo "Make sure you have published a release with 'make publish'"
    exit 1
fi

# Verify it's actually a tar.gz before extracting
FILETYPE=$(file -b "$TMPDIR/${BINARY}.tar.gz" 2>/dev/null || echo "unknown")
case "$FILETYPE" in
    *gzip*) ;;  # OK
    *)
        echo ""
        echo "Error: Downloaded file is not a valid archive (got: $FILETYPE)"
        echo "The release asset might be missing or corrupted."
        exit 1
        ;;
esac

tar -xzf "$TMPDIR/${BINARY}.tar.gz" -C "$TMPDIR"

# Install
if [ -w "$INSTALL_DIR" ]; then
    mv "$TMPDIR/$BINARY" "$INSTALL_DIR/$BINARY"
else
    echo "Need sudo to install to $INSTALL_DIR"
    sudo mv "$TMPDIR/$BINARY" "$INSTALL_DIR/$BINARY"
fi

echo "$BINARY installed to $INSTALL_DIR/$BINARY"
"$INSTALL_DIR/$BINARY" --version 2>/dev/null || true
