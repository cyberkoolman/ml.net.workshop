using Microsoft.ML.Data;
class MeterData
{
    [LoadColumn(0)]
    public string name { get; set; }
    [LoadColumn(1)]
    public DateTime time { get; set; }
    [LoadColumn(2)]
    public float ConsumptionDiffNormalized { get; set; }
}