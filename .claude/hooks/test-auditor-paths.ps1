# PreToolUse hook of the test-auditor subagent: Edit and Write may touch test code only, never app code, snapshots or saved API answers.
# Exit 0 allows the edit; exit 2 blocks it and shows the message to the subagent.
try {
    $path = ([Console]::In.ReadToEnd() | ConvertFrom-Json).tool_input.file_path
    # GetFullPath resolves any "..", so tests/../llm_movies/x.py can't pass as a test file.
    $full = [IO.Path]::GetFullPath([IO.Path]::Combine($env:CLAUDE_PROJECT_DIR, $path)) -replace '\\', '/'
    $root = [IO.Path]::GetFullPath($env:CLAUDE_PROJECT_DIR).TrimEnd('\', '/') -replace '\\', '/'
} catch {
    [Console]::Error.WriteLine('Blocked: could not read the file path.')
    exit 2
}

$relative = $null
if ($full.StartsWith("$root/", [StringComparison]::OrdinalIgnoreCase)) {
    $relative = $full.Substring($root.Length + 1)
}

$testCode = '^(conftest\.py|tests/.+\.py|web/src/.+\.test\.tsx?|web/src/test/[^/]+\.tsx?|web/e2e/[^/]+\.spec\.ts)$'

if ($null -eq $relative -or $relative -notmatch $testCode) {
    [Console]::Error.WriteLine("Blocked: the test auditor edits test code only, not $path. Report the change you wanted instead.")
    exit 2
}
exit 0
