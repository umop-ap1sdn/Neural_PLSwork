package neural_plswork.training;

import neural_plswork.core.NetworkValue;
import neural_plswork.datasets.TrainingDataset;
import neural_plswork.math.Vector;
import neural_plswork.network.Network;

public class ThreadedAgent extends NeuralNetworkTrainer implements Runnable {
    private final MultithreadedTrainer parent;
    private final TrainingDataset td;
    private final int threadID;

    protected Procedure phase;

    public ThreadedAgent(Network nn, MultithreadedTrainer parent, TrainingDataset td, int threadID) {
        super(nn);
        this.parent = parent;
        this.td = td;
        this.threadID = threadID;
        phase = Procedure.TRAIN;
    }

    @Override
    public void train_batch() {
        // Unused
    }

    @Override
    public void train_epoch() {
        int index = 0;
        for(Vector<NetworkValue>[] sample: td) {
            nn.predict(sample[0], threadID);
            nn.calcEvals(sample[1], threadID, index++);
        }

        nn.passEvals(threadID);

    }

    public void adjustWeights() {
        nn.adjustWeights(threadID);
    }

    @Override
    @SuppressWarnings("unchecked")
    public double computeTrainingEval() {
        Vector<NetworkValue>[] y_pred = new Vector[td.length()];
        int y_pred_idx = 0;

        for(Vector<NetworkValue>[] sample: td) {
            y_pred[y_pred_idx++] = nn.predict(sample[0], 0);
        }

        double eval = nn.calculateEval(td.getLabels(), y_pred);

        return eval;
    }

    @Override
    @SuppressWarnings("unchecked")
    public double computeValidationEval() {

        Vector<NetworkValue>[] y_pred = new Vector[td.length()];
        int y_pred_idx = 0;

        for(Vector<NetworkValue>[] sample: td) {
            y_pred[y_pred_idx++] = nn.predict(sample[0], 0);
        }

        double eval = nn.calculateEval(td.getLabels(), y_pred);

        return eval;
    }

    @Override
    public void run() {
        if(phase == Procedure.TRAIN) train_epoch();
        if(phase == Procedure.ADJUST) adjustWeights();
        parent.finish(threadID);

    }

    public void setTrain() {
        phase = Procedure.TRAIN;
    }

    public void setAdjust() {
        phase = Procedure.ADJUST;
    }

    enum Procedure {
        TRAIN,
        ADJUST;
    }
    
}
