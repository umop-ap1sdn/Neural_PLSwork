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
    
    private HashSet<Thread> running;
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
        // finished = new HashSet<>();
        availableThreads = new RollingQueue<>(nn.max_threads() + 1);
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
        
    }

    @Override
    public void train_epoch() {

        int index = 0;
        while(index < agents.length) {
            while(availableThreads.size() > 0 && index < agents.length) {
                agents[index].setTrain(availableThreads.pop());
                threads[index] = new Thread(agents[index]);
                running.add(threads[index]);
                threads[index].start();
                index++;
            }
            
            for(Thread t: running) {
                try {
                    t.join();
                } catch (InterruptedException e) {
                    System.err.println(e.getMessage());
                }
            }

            running.clear();
        }

        index = 0;
        running.clear();

        // System.out.println(availableThreads);
        while(index < agents.length) {
            
            while(availableThreads.size() > 0 && index < agents.length) {
                agents[index].setAdjust(availableThreads.pop());
                threads[index] = new Thread(agents[index]);
                running.add(threads[index]);
                threads[index].start();
                index++;
            }
            
            for(Thread t: running) {
                try {
                    t.join();
                } catch (InterruptedException e) {
                    System.err.println(e.getMessage());
                }
            }

            running.clear();
        }
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
    }
    
}
