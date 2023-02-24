using Microsoft.ML.Data;

public class ModelOutput
{
    [ColumnName("Score")]
    public Single Score { get; set; }
}
