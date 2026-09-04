$f = Get-Item "docs\MTP_MidSem1_Report.html"
$sizeKB = [math]::Round($f.Length / 1KB, 1)
Write-Host "File: $($f.FullName)"
Write-Host "Size: $sizeKB KB"
