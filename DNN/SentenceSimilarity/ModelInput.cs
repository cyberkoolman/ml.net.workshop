using Microsoft.ML.Data;

public class ModelInput
{
    [LoadColumn(2)]
    public string Product { get; set; }

    [LoadColumn(3)]
    public string SearchTerm { get; set; }

    [LoadColumn(4)]
    [ColumnName("Label")]
    public Single Relevance { get; set; }
}
