using System.Collections.Generic;
using CNTK;

namespace CASCPredictor.Functions
{
    /// <summary>
    /// https://github.com/Microsoft/CNTK/blob/master/bindings/python/cntk/layers/blocks.py
    /// </summary>
    internal static class LSTM
    {
        public static Function Build(Variable input, int lstmDimension, DeviceDescriptor device, int cellDimension = 0, bool selfStablize = false)
        {
            if (cellDimension == 0)
                cellDimension = lstmDimension;

            Function lstmFunction = LSTMComponent(input, new[] { lstmDimension }, new[] { cellDimension }, selfStablize, device).h;
            return Function.Alias(lstmFunction, "LSTM");
        }

        private static (Function h, Function c) LSTMComponent(Variable input, NDShape outputShape, NDShape cellShape, bool selfStablize, DeviceDescriptor device)
        {
            var dh = Variable.PlaceholderVariable(outputShape, input.DynamicAxes);
            var dc = Variable.PlaceholderVariable(cellShape, input.DynamicAxes);
            var (h, c) = LSTMCell(input, dh, dc, selfStablize, device);

            h.ReplacePlaceholders(new Dictionary<Variable, Variable> {
                { dh, CNTKLib.PastValue(h) },
                { dc, CNTKLib.PastValue(c) }
            });

            return (h, c);
        }

        private static (Function h, Function c) LSTMCell(Variable input, Variable prevOutput, Variable prevCellState, bool enableSelfStabilization, DeviceDescriptor device)
        {
            int lstmOutputDimension = prevOutput.Shape[0];
            int lstmCellDimension = prevCellState.Shape[0];

            if (enableSelfStabilization)
            {
                prevOutput = Stabilizer.Build(prevOutput, device, "StabilizedPrevOutput");
                prevCellState = Stabilizer.Build(prevCellState, device, "StabilizedPrevCellState");
            }

            uint seed = 1;
            Parameter W = new Parameter(new[] { lstmCellDimension * 4, NDShape.InferredDimension }, DataType.Float, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device, "W");
            Parameter b = new Parameter(new[] { lstmCellDimension * 4 }, DataType.Float, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device, "b");

            Variable linearCombination = CNTKLib.Times(W, input, "linearCombinationInput");
            Variable linearCombinationPlusBias = CNTKLib.Plus(b, linearCombination, "linearCombinationInputPlusBias");

            Parameter H = new Parameter(new[] { lstmCellDimension * 4, lstmOutputDimension }, DataType.Float, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++));
            Variable linearCombinationPrevOutput = CNTKLib.Times(H, prevOutput, "linearCombinationPreviousOutput");

            Variable gateInput = CNTKLib.Plus(linearCombinationPlusBias, linearCombinationPrevOutput, "gateInput");

            // forget-me-not gate
            Variable forgetProjection = CNTKLib.Slice(gateInput, new AxisVector() { new Axis(0) }, new IntVector() { lstmCellDimension * 0 }, new IntVector() { lstmCellDimension * 1 });
            Function ft = CNTKLib.Sigmoid(forgetProjection, "ForgetGate");

            // input gate
            Variable inputProjection = CNTKLib.Slice(gateInput, new AxisVector() { new Axis(0) }, new IntVector() { lstmCellDimension * 1 }, new IntVector() { lstmCellDimension * 2 });
            Function it = CNTKLib.Sigmoid(inputProjection, "InputGate");

            // output gate
            Variable outputProjection = CNTKLib.Slice(gateInput, new AxisVector() { new Axis(0) }, new IntVector() { lstmCellDimension * 2 }, new IntVector() { lstmCellDimension * 3 });
            Function ot = CNTKLib.Sigmoid(outputProjection, "OutputGate");

            // candidate
            Variable candidateProjection = CNTKLib.Slice(gateInput, new AxisVector() { new Axis(0) }, new IntVector() { lstmCellDimension * 3 }, new IntVector() { lstmCellDimension * 4 });
            Function ctt = CNTKLib.Tanh(candidateProjection, "Candidate");

            Function bft = CNTKLib.ElementTimes(prevCellState, ft);
            Function bit = CNTKLib.ElementTimes(it, ctt);
            Function ct = CNTKLib.Plus(bft, bit, "CellState");
            Function ht = CNTKLib.ElementTimes(ot, CNTKLib.Tanh(ct), "Output");

            if (lstmOutputDimension != lstmCellDimension)
            {
                Parameter P = new Parameter(new[] { lstmOutputDimension, lstmCellDimension }, DataType.Float, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++));
                ht = CNTKLib.Times(P, ht, "StandarizedOutput");
            }

            return (ht, ct);
        }
    }
}
