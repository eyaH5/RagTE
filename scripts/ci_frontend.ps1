$ErrorActionPreference = "Stop"

Write-Host "Running Frontend CI Guardrails..." -ForegroundColor Cyan

Write-Host "`n1. Checking Feature Import Boundaries..." -ForegroundColor Yellow
node scripts/check_frontend_boundaries.js

Write-Host "`n2. Checking TypeScript Types..." -ForegroundColor Yellow
cd frontend
npx tsc -b

Write-Host "`n3. Building Production Bundle..." -ForegroundColor Yellow
npm run build

Write-Host "`n✅ All Frontend CI Gates Passed!" -ForegroundColor Green
