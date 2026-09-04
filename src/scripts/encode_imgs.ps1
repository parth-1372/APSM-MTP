$names = @('graph1_bandwidth','graph2_convergence','graph3_semantic_surprise','graph4_per_node_suppression','graph5_mse_boxplot')
for ($i=0; $i -lt $names.Length; $i++) {
    $n = $names[$i]
    $bytes = [System.IO.File]::ReadAllBytes("results\Phase2_Full\$n.png")
    $b64 = [Convert]::ToBase64String($bytes)
    [System.IO.File]::WriteAllText("results\Phase2_Full\img$($i+1).txt", $b64, [System.Text.Encoding]::UTF8)
    Write-Host "Encoded $n"
}
Write-Host "All done"
