#!/bin/sh
# Sourced bootstrap primitives; acquiring uv never requires a valid Python lock.

bootstrap_checksum() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d ' ' -f 1
    else
        echo "bootstrap: requires sha256sum or shasum" >&2
        return 1
    fi
}

# Call only inside bootstrap subshells. Native commands retain caller settings.
bootstrap_uv_sanitize() {
    unset VIRTUAL_ENV PYTHONHOME PYTHONPATH PYTHONSAFEPATH __PYVENV_LAUNCHER__
    for variable in $(env | sed -n 's/^\(UV_[A-Z0-9_]*\)=.*/\1/p'); do
        case "$variable" in
            UV_CACHE_DIR|UV_PYTHON_CACHE_DIR|UV_HTTP_TIMEOUT|UV_HTTP_CONNECT_TIMEOUT|UV_HTTP_RETRIES|UV_SYSTEM_CERTS|UV_NATIVE_TLS) ;;
            *) unset "$variable" ;;
        esac
    done
}

# Emit only the verified executable path; all diagnostics belong on stderr.
bootstrap_uv() (
    bootstrap_root=$1
    entrypoint=${2:-python}
    fail() { echo "$entrypoint: $*" >&2; exit 1; }
    case "${FRONTIER_BOOTSTRAP_OFFLINE-0}" in
        0) offline=false ;;
        1) offline=true ;;
        *) fail "FRONTIER_BOOTSTRAP_OFFLINE must be 0 or 1; see README.md" ;;
    esac
    case "$(uname -s)" in
        Darwin) system=darwin ;;
        Linux) system=linux ;;
        *) fail "unsupported OS; see README.md" ;;
    esac
    case "$(uname -m)" in
        arm64|aarch64) arch=arm64 ;;
        x86_64|amd64) arch=amd64 ;;
        *) fail "unsupported architecture; see README.md" ;;
    esac
    host="$system-$arch"
    version='' url='' digest='' member=''
    while read -r row_host row_version row_url row_digest row_member; do
        if [ "$row_host" = "$host" ]; then
            version=$row_version url=$row_url digest=$row_digest member=$row_member
            break
        fi
    done < "$bootstrap_root/config/bootstrap-tools.lock"
    [ -n "$version" ] || fail "no bootstrap tool pinned for $host"
    cache=${FRONTIER_TOOL_CACHE:-${XDG_CACHE_HOME:-"$HOME/.cache"}/frontier-preview-tools}
    case "$cache" in /*) ;; *) fail "FRONTIER_TOOL_CACHE must be an absolute path" ;; esac
    archive="$cache/downloads/$digest"
    directory="$cache/uv/$version/$host/$digest"
    uv="$directory/uv"

    if [ ! -x "$uv" ] || [ ! -f "$directory/uv.sha256" ] || \
            [ "$(bootstrap_checksum "$uv")" != "$(cat "$directory/uv.sha256")" ]; then
        mkdir -p "$cache/downloads" "$directory"
        stage=$(mktemp -d "$directory/.bootstrap.XXXXXX")
        trap 'rm -rf "$stage"' EXIT HUP INT TERM
        if [ ! -f "$archive" ] || [ "$(bootstrap_checksum "$archive")" != "$digest" ]; then
            [ "$offline" = false ] || fail "uv $version is not cached; rerun ./$entrypoint online"
            command -v curl >/dev/null 2>&1 || fail "bootstrap requires curl"
            echo "Downloading uv $version for $host" >&2
            curl --fail --location --silent --show-error --proto '=https' --tlsv1.2 \
                --retry 3 --connect-timeout 30 --max-time 300 --output "$stage/archive" "$url"
            [ "$(bootstrap_checksum "$stage/archive")" = "$digest" ] || fail "uv SHA-256 mismatch; refusing to execute download"
            mv "$stage/archive" "$archive"
        fi
        tar -xzf "$archive" -C "$stage" "$member"
        chmod 755 "$stage/$member"
        bootstrap_checksum "$stage/$member" > "$stage/uv.sha256"
        # Concurrent bootstraps install the same verified bytes atomically.
        mv "$stage/$member" "$uv"
        mv "$stage/uv.sha256" "$directory/uv.sha256"
        rm -rf "$stage"
        trap - EXIT HUP INT TERM
    fi
    bootstrap_uv_sanitize
    case "$("$uv" --version)" in
        "uv $version"|"uv $version ("*) ;;
        *) fail "cached uv has the wrong version; remove $directory and retry" ;;
    esac
    printf '%s\n' "$uv"
)
