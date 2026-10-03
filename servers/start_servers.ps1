# Start all small model inference servers.
$workspace = Split-Path -Parent $PSScriptRoot
Push-Location $workspace

Write-Host "Starting SAGE small model servers..."

Start-Process -WindowStyle Hidden -FilePath python -ArgumentList "-m", "servers.human_detect_server"
Write-Host "  human_detect -> :8001"

Start-Process -WindowStyle Hidden -FilePath python -ArgumentList "-m", "servers.pose_server"
Write-Host "  pose -> :8002"

Start-Process -WindowStyle Hidden -FilePath python -ArgumentList "-m", "servers.depth_server"
Write-Host "  depth -> :8003"

Start-Process -WindowStyle Hidden -FilePath python -ArgumentList "-m", "servers.face_landmark_server"
Write-Host "  face_landmark -> :8005"

Pop-Location
Write-Host "All servers started."
