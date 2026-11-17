# push_to_github.ps1 — Helper script to link remote and push to GitHub
param(
    [Parameter(Mandatory=$false)]
    [string]$RepoUrl
)

if (-not $RepoUrl) {
    $RepoUrl = Read-Host "Enter your GitHub repository URL (e.g. https://github.com/<username>/missing-children-ml-india.git)"
}

if ($RepoUrl) {
    git remote remove origin 2>$null
    git remote add origin $RepoUrl
    Write-Host "Configured remote origin to $RepoUrl" -ForegroundColor Green
    Write-Host "Pushing to GitHub (main branch)..." -ForegroundColor Cyan
    git push -u origin main
} else {
    Write-Host "No URL entered. Run again with your repository URL." -ForegroundColor Yellow
}
