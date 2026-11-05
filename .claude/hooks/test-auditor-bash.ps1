# PreToolUse hook of the test-auditor subagent: only read-only git, the test commands and removing its own throwaway mutant tests may run.
# Exit 0 allows the command; exit 2 blocks it and shows the message to the subagent.
try {
    $command = ([Console]::In.ReadToEnd() | ConvertFrom-Json).tool_input.command
} catch {
    [Console]::Error.WriteLine('Blocked: could not read the command.')
    exit 2
}

$allowed = @(
    '^\s*git\s+(status|diff|log|show|blame|ls-files|rev-parse|merge-base)(\s|$)',
    '^\s*uv\s+run\s+pytest(\s|$)',
    '^\s*npm\s+--prefix\s+web\s+(test(\s+--\s+[\w./-]+)?|run\s+(check|coverage))\s*$',
    '^\s*rm\s+(tests/test_zz_mutant_\w+\.py|web/src/zz_mutant_\w+\.test\.tsx?)\s*$'
)
$shellSyntax = '[;&|<>`$()\r\n]'
# git: --output writes a file and --ext-diff runs a program, and git accepts them abbreviated.
# pytest: --basetemp wipes the folder it is given; the others load other code or settings.
$unsafeOption = '\s(--ou|--ex|--basetemp|--rootdir|--confcutdir|--override-ini|-c|-p|-o)(\s|=|$)|\s--(ou|ex)'

$isAllowed = $false
foreach ($pattern in $allowed) {
    if ($command -match $pattern) { $isAllowed = $true }
}

if (-not $isAllowed -or $command -match $shellSyntax -or $command -match $unsafeOption) {
    [Console]::Error.WriteLine('Blocked: the test auditor may only run one plain read-only git command, uv run pytest, npm --prefix web test|run check|run coverage, or rm of its own zz_mutant test, with no pipes or redirects.')
    exit 2
}
exit 0
