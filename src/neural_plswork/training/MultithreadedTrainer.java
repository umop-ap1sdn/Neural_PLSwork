package neural_plswork.training;

import java.util.HashSet;

import neural_plswork.datasets.BatchedTrainingDataset;
import neural_plswork.datasets.TrainingDataset;
import neural_plswork.network.Network;

public class MultithreadedTrainer extends NeuralNetworkTrainer {

    private final BatchedTrainingDataset train_set;
    private final BatchedTrainingDataset test_set;
    private final ThreadedAgent[] agents;
    private final ThreadedAgent[] test_agents;
    private final Thread[] threads;
    private int index = 0;

    private HashSet<Integer> running;
    private HashSet<Integer> finished;

    public MultithreadedTrainer(Network nn, BatchedTrainingDataset train_set, BatchedTrainingDataset test_set) {
        super(nn);
        this.train_set = train_set;
        this.test_set = test_set;
        this.agents = new ThreadedAgent[train_set.batch_num()];
        this.test_agents = new ThreadedAgent[test_set.batch_num()];
        this.threads = new Thread[train_set.batch_num()];

        running = new HashSet<>();
        finished = new HashSet<>();
        prepareThreads();
    }

    private void prepareThreads() {
        int index = 0;
        for(TrainingDataset td: train_set) {
            ThreadedAgent ta = new ThreadedAgent(nn, this, td, index);
            agents[index] = ta;
            threads[index++] = new Thread(ta);
        }

        for(TrainingDataset td: test_set) {
            ThreadedAgent ta = new ThreadedAgent(nn, this, td, index++);
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
        double eval = 0;
        for(ThreadedAgent ta: agents) {
            eval += ta.computeTrainingEval();
        }
        return eval;
    }

    @Override
    public double computeValidationEval() {
        double eval = 0;
        for(ThreadedAgent ta: test_agents) {
            eval += ta.computeValidationEval();
        }

        return eval;
    }

    protected void finish(int threadID) {
        running.remove(threadID);
        finished.add(threadID);
    }
    
}
