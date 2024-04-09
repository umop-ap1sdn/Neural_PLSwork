package neural_plswork.training;

import java.util.HashSet;

import neural_plswork.datasets.BatchedTrainingDataset;
import neural_plswork.datasets.TrainingDataset;
import neural_plswork.network.Network;

public class MultithreadedTrainer extends NeuralNetworkTrainer {

    private final BatchedTrainingDataset btd;
    private final ThreadedAgent[] agents;
    private final Thread[] threads;
    private int index = 0;

    private HashSet<Integer> running;
    private HashSet<Integer> finished;

    public MultithreadedTrainer(Network nn, BatchedTrainingDataset btd) {
        super(nn);
        this.btd = btd;
        this.agents = new ThreadedAgent[btd.batch_num()];
        this.threads = new Thread[btd.batch_num()];

        running = new HashSet<>();
        finished = new HashSet<>();
        prepareThreads();
    }

    private void prepareThreads() {
        int index = 0;
        for(TrainingDataset td: btd) {
            ThreadedAgent ta = new ThreadedAgent(nn, this, td, index);
            agents[index] = ta;
            threads[index++] = new Thread(ta);
        }
    }

    @Override
    public void train_batch() {
        if(running.size() >= nn.max_threads()) return;
        agents[index].setTrain();
        threads[index].run();
        running.add(index);
        while(!finished.contains(index));
        finished.remove(index);
        agents[index].setAdjust();
        threads[index].run();
        running.add(index);
        while(!finished.contains(index));
        finished.remove(index);
        index = (index + 1) % agents.length;
    }

    @Override
    public void train_epoch() {
        running.clear();
        finished.clear();
        int index = 0;
        while(finished.size() < agents.length) {
            while(running.size() < nn.max_threads() && index < agents.length) {
                agents[index].setTrain();
                threads[index].run();
                running.add(index++);
            }
        }

        running.clear();
        finished.clear();

        while(finished.size() < agents.length) {
            while(running.size() < nn.max_threads() && index < agents.length) {
                agents[index].setAdjust();
                threads[index].run();
                running.add(index++);
            }
        }

        running.clear();
        finished.clear();
    }

    @Override
    public double computeTrainingEval() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'computeTrainingEval'");
    }

    @Override
    public double computeValidationEval() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'computeValidationEval'");
    }

    protected void finish(int threadID) {
        running.remove(threadID);
        finished.add(threadID);
    }
    
}
