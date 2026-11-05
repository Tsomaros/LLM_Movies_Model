# PreToolUse hook of the code-reviewer subagent: only plain read-only git commands may run.
# Exit 0 allows the command; exit 2 blocks it and shows the message to the subagent.
try {
    $command = ([Console]::In.ReadToEnd() | ConvertFrom-Json).tool_input.command
} catch {
    [Console]::Error.WriteLine('Blocked: could not read the command.')
    exit 2
}

$readOnly = '^\s*git\s+(status|diff|log|show|blame|ls-files|rev-parse|merge-base)(\s|$)'
$shellSyntax = '[;&|<>`$()\r\n]'
# --output writes a file and --ext-diff runs a program; git also accepts these options abbreviated.
$unsafeOption = '\s--(ou|ex)'

if ($command -notmatch $readOnly -or $command -match $shellSyntax -or $command -match $unsafeOption) {
    [Console]::Error.WriteLine('Blocked: the reviewer may only run one plain git status, diff, log, show, blame, ls-files, rev-parse or merge-base command, with no pipes or redirects.')
    exit 2
}
exit 0
