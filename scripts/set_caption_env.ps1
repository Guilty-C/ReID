$ErrorActionPreference = 'Stop'
# Load local API config and export CAPTION_* env vars (supports two schemas)
$cfgPath = Join-Path (Get-Location) 'configs/caption_api.json'
if (-not (Test-Path $cfgPath)) {
  Write-Error "Config not found: $cfgPath"
  exit 1
}
$cfgRaw = Get-Content -Raw -Path $cfgPath
$cfg = $cfgRaw | ConvertFrom-Json

# Read URL/KEY/MODEL from either CAPTION_* keys or plain keys
$baseUrl = $null
$key = $null
$model = $null
if ($cfg.PSObject.Properties.Name -contains 'CAPTION_API_URL') { $baseUrl = $cfg.CAPTION_API_URL } elseif ($cfg.PSObject.Properties.Name -contains 'url') { $baseUrl = $cfg.url }
if ($cfg.PSObject.Properties.Name -contains 'CAPTION_API_KEY') { $key = $cfg.CAPTION_API_KEY } elseif ($cfg.PSObject.Properties.Name -contains 'key') { $key = $cfg.key }
if ($cfg.PSObject.Properties.Name -contains 'CAPTION_API_MODEL') { $model = $cfg.CAPTION_API_MODEL } elseif ($cfg.PSObject.Properties.Name -contains 'model') { $model = $cfg.model }

# Normalize endpoint to OpenAI-compatible chat completions if needed
$baseUrl = ($baseUrl).TrimEnd('/')
if ($baseUrl -match '/chat/completions$' -or $baseUrl -match '/responses$' -or $baseUrl -match '/v1/messages$') {
  $endpoint = $baseUrl
} else {
  $endpoint = "$baseUrl/v1/chat/completions"
}

# Export env vars
$env:CAPTION_API_URL = $endpoint
$env:CAPTION_API_KEY = $key
if ($null -ne $model -and $model -ne '') { $env:CAPTION_API_MODEL = $model } else { $env:CAPTION_API_MODEL = 'gpt-4o-mini' }

Write-Host "CAPTION_API_URL=$($env:CAPTION_API_URL)"
Write-Host "CAPTION_API_MODEL=$($env:CAPTION_API_MODEL)"
Write-Host "CAPTION_API_KEY set? $([bool]$env:CAPTION_API_KEY)"