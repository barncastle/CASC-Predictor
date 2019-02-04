using CNTK;

namespace CASCPredictor.Functions
{
    /// <summary>
    /// https://github.com/Microsoft/CNTK/blob/master/Examples/TrainingCSharp/Common/LSTMSequenceClassifier.cs
    /// </summary>
    internal static class Stabilizer
    {
        public static Function Build(Variable input, DeviceDescriptor device, string outputName = "Stabilizer")
        {
            Constant f = Constant.Scalar(4.0f, device);
            Constant fInv = Constant.Scalar(DataType.Float, 1.0 / 4.0f);

            // 0.99537863 == 1/f*ln (e^f-1)
            Function alpha = Constant.Scalar(DataType.Float, 1.0f) + CNTKLib.Exp(CNTKLib.ElementTimes(f, new Parameter(new NDShape(), DataType.Float, 0.99537863, device, "alpha")));
            Function beta = CNTKLib.ElementTimes(fInv, CNTKLib.Log(alpha), "beta");

            return Function.Alias(CNTKLib.ElementTimes(beta, input), outputName);
        }
    }
}
