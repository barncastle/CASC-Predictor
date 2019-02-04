using System.Diagnostics;
using System.Linq;
using CNTK;

namespace CASCPredictor.Functions
{
    /// <summary>
    /// https://github.com/Microsoft/CNTK/blob/master/Examples/TrainingCSharp/Common/TestHelper.cs
    /// </summary>
    internal static class Dense
    {
        public static Function Build(Variable input, int outputDimension, DeviceDescriptor device, Activation activation = Activation.None)
        {
            if (input.Shape.Rank != 1)
            {
                int newDimension = input.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                input = CNTKLib.Reshape(input, new int[] { newDimension });
            }

            Function fclLayer = FullyConnectedLinearLayer(input, outputDimension, device);

            Function dense;
            switch (activation)
            {
                case Activation.ReLU:
                    dense = CNTKLib.ReLU(fclLayer);
                    break;
                case Activation.Sigmoid:
                    dense = CNTKLib.Sigmoid(fclLayer);
                    break;
                case Activation.Tanh:
                    dense = CNTKLib.Tanh(fclLayer);
                    break;
                default:
                    dense = fclLayer;
                    break;
            }

            return Function.Alias(dense, "Dense");
        }

        private static Function FullyConnectedLinearLayer(Variable input, int outputDimension, DeviceDescriptor device, string outputName = "")
        {
            Debug.Assert(input.Shape.Rank == 1);

            var initializer = CNTKLib.GlorotUniformInitializer(CNTKLib.DefaultParamInitScale, CNTKLib.SentinelValueForInferParamInitRank, CNTKLib.SentinelValueForInferParamInitRank, 1);
            var timesParam = new Parameter(new[] { outputDimension, input.Shape[0] }, DataType.Float, initializer, device, "timesParam");
            var timesFunction = CNTKLib.Times(timesParam, input, "times");
            var plusParam = new Parameter(new[] { outputDimension }, 0.0f, device, "plusParam");

            return CNTKLib.Plus(plusParam, timesFunction, outputName);
        }
    }

    internal enum Activation
    {
        None,
        ReLU,
        Sigmoid,
        Tanh
    }
}
