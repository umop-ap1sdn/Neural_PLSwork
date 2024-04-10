package neural_plswork.training;

import java.util.HashSet;

import neural_plswork.datasets.BatchedTrainingDataset;
import neural_plswork.datasets.TrainingDataset;
import neural_plswork.network.Network;
import neural_plswork.rollingqueue.RollingQueue;

public class MultithreadedTrainer extends NeuralNetworkTrainer {

    private final BatchedTrainingDataset train_set;
    private final BatchedTrainingDataset test_set;
    private final ThreadedAgent[] agents;
    private final ThreadedAgent[] test_agents;
    private final Thread[] threads;
    private int index = 0;

    private HashSet<Integer> running;
    private HashSet<Integer> finished;
    private RollingQueue<Integer> availableThreads;

    public MultithreadedTrainer(Network nn, BatchedTrainingDataset train_set, BatchedTrainingDataset test_set) {
        super(nn);
        this.train_set = train_set;
        this.test_set = test_set;
        this.agents = new ThreadedAgent[train_set.batch_num()];
        if(test_set == null) this.test_agents = new ThreadedAgent[0];
        else this.test_agents = new ThreadedAgent[test_set.batch_num()];
        this.threads = new Thread[train_set.batch_num()];

        running = new HashSet<>();
        finished = new HashSet<>();
        availableThreads = new RollingQueue<>(nn.max_threads());
        prepareThreads();
    }

    private void prepareThreads() {
        int index = 0;
        for(TrainingDataset td: train_set) {
            ThreadedAgent ta = new ThreadedAgent(nn, this, td, index);
            agents[index] = ta;
            threads[index++] = new Thread(ta);
        }

        index = 0;
        if(test_set != null) {
            for(TrainingDataset td: test_set) {
                ThreadedAgent ta = new ThreadedAgent(nn, this, td, index);
                test_agents[index++] = ta;
            }
        }

        for(int i = 0; i < nn.max_threads(); i++) {
            availableThreads.push(i);
        }
    }

    @Override
    public void train_batch() {
        if(running.size() >= nn.max_threads()) return;
        agents[index].setTrain(0);
        threads[index].run();
        running.add(index);
        while(!finished.contains(index));
        finished.remove(index);
        agents[index].setAdjust(0);
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
            while(availableThreads.size() > 0 && index < agents.length) {
                running.add(index);
                agents[index].setTrain(availableThreads.pop());
                threads[index] = new Thread(agents[index]);
                threads[index++].start();
                
            }
            //if(index >= agents.length) System.out.println(finished.size());
        }

        running.clear();
        finished.clear();

        index = 0;

        while(finished.size() < agents.length) {
            while(availableThreads.size() > 0 && index < agents.length) {
                running.add(index);
                agents[index].setAdjust(availableThreads.pop());
                threads[index] = new Thread(agents[index]);
                threads[index++].start();
            
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

    protected synchronized void finish(int threadID, int threadIndex) {
        availableThreads.push(threadIndex);
        
        running.remove(threadID);
        finished.add(threadID);

    }
    
}
