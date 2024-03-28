package neural_plswork.core;

import neural_plswork.initialize.Initializer;
import neural_plswork.layers.basic.InputLayer;
import neural_plswork.layers.basic.OutputLayer;
import neural_plswork.math.Matrix;
import neural_plswork.math.MatrixElement;
import neural_plswork.math.Vector;
import neural_plswork.optimizer.OptimizationFunction;
import neural_plswork.regularization.penalize.None;
import neural_plswork.regularization.penalize.Penalty;

public class ConnectionLayer {
    
    Matrix<NetworkValue> primaryLayer;
    Vector<NetworkValue> biasVector;

    private NeuronLayer srcLayer;
    private NeuronLayer destLayer;

    private Initializer initializer;

    private OptimizationFunction primaryOptimizer;
    private OptimizationFunction biasOptimizer;
    
    
    private Penalty penalty;

    public ConnectionLayer(NeuronLayer srcLayer, NeuronLayer destLayer, Initializer initializer, OptimizationFunction optimizer, Penalty penalty) throws InvalidNetworkConstructionException {
        if(srcLayer instanceof OutputLayer) 
            throw new InvalidNetworkConstructionException("Cannot use Output Layer as Source Layer");
        if(destLayer instanceof InputLayer)
            throw new InvalidNetworkConstructionException("Cannot use Input Layer as Dest Layer");
        
        this.srcLayer = srcLayer;
        this.destLayer = destLayer;
        this.initializer = initializer;

        this.primaryOptimizer = optimizer;
        this.biasOptimizer = optimizer.copy();
        this.penalty = penalty;

        this.initialize();

    }

    public ConnectionLayer(NeuronLayer srcLayer, NeuronLayer destLayer, Matrix<NetworkValue> primaryLayer, Vector<NetworkValue> biasVector, OptimizationFunction optimizer, Penalty penalty) throws InvalidNetworkConstructionException {
        if(srcLayer instanceof OutputLayer) 
            throw new InvalidNetworkConstructionException("Cannot use Output Layer as Source Layer");
        if(destLayer instanceof InputLayer)
            throw new InvalidNetworkConstructionException("Cannot use Input Layer as Dest Layer");
        
        this.srcLayer = srcLayer;
        this.destLayer = destLayer;
        
        this.primaryLayer = primaryLayer;
        this.biasVector = biasVector;

        this.primaryOptimizer = optimizer;
        this.biasOptimizer = optimizer.copy();
        this.penalty = penalty;
    }

    public ConnectionLayer(NeuronLayer srcLayer, NeuronLayer destLayer, NetworkValue[][] primaryLayer, NetworkValue[] biasVector, OptimizationFunction optimizer, Penalty penalty) throws InvalidNetworkConstructionException {
        if(srcLayer instanceof OutputLayer) 
            throw new InvalidNetworkConstructionException("Cannot use Output Layer as Source Layer");
        if(destLayer instanceof InputLayer)
            throw new InvalidNetworkConstructionException("Cannot use Input Layer as Dest Layer");
        
        this.srcLayer = srcLayer;
        this.destLayer = destLayer;
        
        this.primaryLayer = new Matrix<>(primaryLayer);
        this.biasVector = new Vector<>(biasVector);

        this.primaryOptimizer = optimizer;
        this.biasOptimizer = optimizer.copy();
        this.penalty = penalty;
    }

    private void initialize() {
        int rows = destLayer.size();
        int cols = srcLayer.size();

        NetworkValue[][] matrix = new NetworkValue[rows][cols];

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                matrix[i][j] = initializer.getNextWeight(i, j);
            }
        }

        this.primaryLayer = new Matrix<>(matrix);

        NetworkValue[] vector = new NetworkValue[rows];

        for(int i = 0; i < rows; i++) {
            if(srcLayer.getBias()) vector[i] = initializer.getNextWeight(i, 0);
            else vector[i] = new NetworkValue();
        }

        this.biasVector = new Vector<>(vector);
    }

    public Vector<NetworkValue> forwardPass(Vector<NetworkValue> input) {
        Matrix<NetworkValue> multiplied = primaryLayer.multiply(input);
        if(srcLayer.getBias()) multiplied = multiplied.add(biasVector);
        return multiplied.getAsVector();
    }

    public Vector<NetworkValue> forwardPass(int thread) {
        return forwardPass(srcLayer.getRecentValues(thread));
    }

    public void adjustWeights(double learning_rate, int steps, boolean descending, int thread) {
        for(int time = 0; time < steps; time++) {
            Matrix<NetworkValue> transposed = srcLayer.getValues(time, thread).transpose();
            
            Matrix<NetworkValue> primaryGradients = destLayer.getEval(time, thread).multiply(transposed);
            Vector<NetworkValue> biasGradients = destLayer.getEval(time, thread).copy();
            
            if(!(penalty instanceof None)) {
                Matrix<NetworkValue> primaryPenalty = penalty.getDerivative(primaryLayer);
                Vector<NetworkValue> biasPenalty = penalty.getDerivative(biasVector).getAsVector();
                if(!descending) {
                    primaryPenalty = primaryPenalty.scale(new NetworkValue(-1.0));
                    biasPenalty = biasPenalty.<NetworkValue, MatrixElement>scale(new NetworkValue(-1.0)).getAsVector();
                }

                
                primaryGradients = primaryGradients.add(primaryPenalty);
                biasGradients = biasGradients.<NetworkValue, NetworkValue>add(biasPenalty).getAsVector();
            }

            Matrix<NetworkValue> primaryDeltas = primaryOptimizer.computeDeltas(primaryGradients, learning_rate);
            Vector<NetworkValue> biasDeltas = biasOptimizer.computeDeltas(biasGradients, learning_rate).getAsVector();

            if(descending) {
                primaryDeltas = primaryDeltas.scale(new NetworkValue(-1.0));
                biasDeltas = biasDeltas.<NetworkValue, MatrixElement>scale(new NetworkValue(-1.0)).getAsVector();
            }

            primaryLayer = primaryLayer.add(primaryDeltas);
            if(srcLayer.getBias()) biasVector = biasVector.<NetworkValue, NetworkValue>add(biasDeltas).getAsVector();

        }
    }

    public void setOptimizer(OptimizationFunction optimizer) {
        this.primaryOptimizer = optimizer;
        this.biasOptimizer = optimizer.copy();
    }

    public void setPenalty(Penalty penalty) {
        this.penalty = penalty;
    }

    public Matrix<NetworkValue> getLayer() {
        return this.primaryLayer;
    }
}
