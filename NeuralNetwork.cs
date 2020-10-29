using System;
using System.Collections.Generic;
using System.Linq;

namespace Kara.Modules
{
    public class Layer
    {
        public List<Neuron> Neurons { get; set; }
    }
    public class LayerInput : Layer
    {
    }
    public class LayerHidden : Layer
    {
    }
    public class LayerOutput : Layer
    {
    }
    public class Neuron
    {
        public List<Connection> Connections1 { get; set; }
        public List<Connection> Connections2 { get; set; }
        public float InputValue { get; set; }
        public float OutputValue { get; set; }
        public float Bias { get; set; }
        public float Error { get; set; }
        public float Gradient { get; set; }
    }
    public class NeuronInput : Neuron
    {
    }
    public class NeuronHidden : Neuron
    {
    }
    public class NeuronOutput : Neuron
    {
    }
    public class Connection
    {
        public Neuron Neuron1 { get; set; }
        public Neuron Neuron2 { get; set; }
        public float Weight { get; set; }
        public float Gradient { get; set; }
    }
    public class NeuralNetwork
    {
        private int Epoch { get; set; }
        private float Error { get; set; }
        private float ErrorApproximation { get; set; }
        private List<Layer> Layers { get; }

        private NeuralNetwork()
        {
            this.Epoch = 0;
            this.Error = 1;
            this.ErrorApproximation = 1;
            this.Layers = new List<Layer>();
        }

        private void AddLayerInput(int neurons)
        {
            var layer = new LayerInput
            {
                Neurons = new List<Neuron>(),
            };
            for (var i = 1; i <= neurons; i++)
            {
                var neuron = new NeuronInput
                {
                    Connections1 = new List<Connection>(),
                    Connections2 = new List<Connection>(),
                };
                neuron.Bias = Math2.Range(-1f, 1);
                layer.Neurons.Add(neuron);
            }
            this.Layers.Add(layer);
        }

        private void AddLayerHidden(int neurons)
        {
            var layer = new LayerHidden
            {
                Neurons = new List<Neuron>(),
            };
            for (var i = 1; i <= neurons; i++)
            {
                var neuron = new NeuronHidden
                {
                    Connections1 = new List<Connection>(),
                    Connections2 = new List<Connection>(),
                };
                neuron.Bias = Math2.Range(-1f, 1);
                layer.Neurons.Add(neuron);
            }
            this.Layers.Add(layer);
        }

        private void AddLayerOutput(int neurons)
        {
            var layer = new LayerOutput
            {
                Neurons = new List<Neuron>(),
            };
            for (var i = 1; i <= neurons; i++)
            {
                var neuron = new NeuronOutput
                {
                    Connections1 = new List<Connection>(),
                    Connections2 = new List<Connection>(),
                };
                neuron.Bias = Math2.Range(-1f, 1);
                layer.Neurons.Add(neuron);
            }
            this.Layers.Add(layer);
        }

        private void ConnectLayers()
        {
            var layer1 = default(Layer);
            foreach (var layer2 in this.Layers)
            {
                if (layer1 != null)
                {
                    foreach (var neuron1 in layer1.Neurons)
                    {
                        foreach (var neuron2 in layer2.Neurons)
                        {
                            var connection = new Connection
                            {
                                Neuron1 = neuron1,
                                Neuron2 = neuron2,
                            };
                            connection.Neuron1.Connections2.Add(connection);
                            connection.Neuron2.Connections1.Add(connection);
                            connection.Weight = Math2.Range(-1f, 1);
                        }
                    }
                }
                layer1 = layer2;
            }
        }

        public static NeuralNetwork Generate(int inputLength, int outputLength)
        {
            var hiddenLayers = Math2.RoundToInt(inputLength * (2f / 3) + outputLength);
            hiddenLayers = Math2.Clamp(hiddenLayers, inputLength, outputLength);
            hiddenLayers = Math2.Clamp(hiddenLayers, 0, inputLength * 2 - 1);

            var network = new NeuralNetwork();
            network.AddLayerInput(inputLength);
            network.AddLayerHidden(hiddenLayers);
            network.AddLayerHidden(hiddenLayers);
            network.AddLayerOutput(outputLength);
            network.ConnectLayers();
            return network;
        }

        public void Train(List<float> input, List<float> output)
        {
            var learningRate = this.Error;
            var errors = new List<float>();

            // calculate values
            foreach (var layer in this.Layers.Take(1))
            {
                foreach (var neuron in layer.Neurons)
                {
                    var index = layer.Neurons.IndexOf(neuron);
                    var value = input.ElementAt(index);
                    neuron.OutputValue = value;
                }
            }
            foreach (var layer in this.Layers.Skip(1))
            {
                foreach (var neuron in layer.Neurons)
                {
                    neuron.InputValue = neuron.Connections1.Sum(x => x.Neuron1.OutputValue * x.Weight) + neuron.Bias;
                    neuron.OutputValue = Math2.Sigmoid(neuron.InputValue);
                }
            }

            // calculate error
            foreach (var layer in this.Layers.AsEnumerable().Reverse().Take(1))
            {
                foreach (var neuron in layer.Neurons)
                {
                    var index = layer.Neurons.IndexOf(neuron);
                    var value = output.ElementAt(index);
                    neuron.Error = value - neuron.OutputValue;
                    errors.Add(neuron.Error);
                }
            }

            // calculate neuron gradient
            foreach (var layer in this.Layers.AsEnumerable().Reverse().Take(1))
            {
                foreach (var neuron in layer.Neurons)
                {
                    var delta = Math2.SigmoidDerivative(neuron.InputValue) * neuron.Error;
                    neuron.Gradient = delta * learningRate;
                }
            }
            foreach (var layer in this.Layers.AsEnumerable().Reverse().Skip(1))
            {
                foreach (var neuron in layer.Neurons)
                {
                    var delta = Math2.SigmoidDerivative(neuron.InputValue) * neuron.Connections2.Sum(x => x.Neuron2.Gradient * x.Weight);
                    neuron.Gradient = delta * learningRate;
                }
            }

            // calculate connection gradient
            foreach (var layer in this.Layers.Skip(1).Reverse())
            {
                foreach (var neuron in layer.Neurons)
                {
                    foreach (var connection in neuron.Connections1)
                    {
                        var delta = connection.Neuron1.OutputValue * connection.Neuron2.Gradient;
                        connection.Gradient = delta * learningRate;
                    }
                }
            }

            // calculate bias
            foreach (var layer in this.Layers)
            {
                foreach (var neuron in layer.Neurons)
                {
                    neuron.Bias += neuron.Gradient;
                }
            }

            // calculate weight
            foreach (var layer in this.Layers.Skip(1))
            {
                foreach (var neuron in layer.Neurons)
                {
                    foreach (var connection in neuron.Connections2)
                    {
                        connection.Weight += connection.Gradient;
                    }
                }
            }

            this.Epoch++;
            this.Error = Math.Abs(errors.Sum());
            this.ErrorApproximation = this.ErrorApproximation * .9f + this.Error * .1f;
        }

        public List<float> Fire(List<float> input)
        {
            var result = new List<float>();

            foreach (var layer in this.Layers.Take(1))
            {
                foreach (var neuron in layer.Neurons)
                {
                    var index = layer.Neurons.IndexOf(neuron);
                    var value = input.ElementAt(index);
                    neuron.OutputValue = value;
                }
            }
            foreach (var layer in this.Layers.Skip(1))
            {
                foreach (var neuron in layer.Neurons)
                {
                    neuron.InputValue = neuron.Connections1.Sum(x => x.Neuron1.OutputValue * x.Weight) + neuron.Bias;
                    neuron.OutputValue = Math2.Sigmoid(neuron.InputValue);
                }
            }
            foreach (var layer in this.Layers.AsEnumerable().Reverse().Take(1))
            {
                foreach (var neuron in layer.Neurons)
                {
                    result.Add(neuron.OutputValue);
                }
            }

            return result;
        }

        public override string ToString()
        {
            return new List<string>
            {
                $"EPOCH: {this.Epoch}",
                $"ERROR: {this.Error:N2}",
                $"ERROR: {this.ErrorApproximation * 100:N0}%",
                string.Empty,
                this.Layers.Select(x => this.ToString(x)).Join("\n\n"),
            }.Join("\n");
        }

        private string ToString(Layer layer)
        {
            return new List<string>
            {
                layer.Neurons.Select(x => this.ToString(x)).Join("\n"),
            }.Join("\n");
        }

        private string ToString(Neuron neuron)
        {
            return new List<string>
            {
                $"IN: {neuron.InputValue:N2}",
                $"OUT: {neuron.OutputValue:N2}",
                $"BIA: {neuron.Bias:N2}",
                $"W1: {neuron.Connections1.Select(x => x.Weight).DefaultIfEmpty().Average():N2}",
                $"W2: {neuron.Connections2.Select(x => x.Weight).DefaultIfEmpty().Average():N2}",
            }.Join("\t");
        }
    }
}