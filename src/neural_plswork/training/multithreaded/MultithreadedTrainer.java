package neural_plswork.training.multithreaded;

import neural_plswork.datasets.BatchedTrainingDataset;
import neural_plswork.datasets.TrainTestSplit;
import neural_plswork.datasets.TrainingDataset;
import neural_plswork.network.Network;
import neural_plswork.rollingqueue.RollingQueue;
import neural_plswork.training.NeuralNetworkTrainer;

public class MultithreadedTrainer extends NeuralNetworkTrainer {

    private final BatchedTrainingDataset train_set;
    private final BatchedTrainingDataset test_set;
    private final ThreadedAgent[] agents;
    private final ThreadedAgent[] test_agents;
    private final Thread[] threads;
    
    private RollingQueue<Integer> availableThreads;
    private RollingQueue<Thread> ready;

    int index = 0;

    private final double DEFAULT_TRAINING_RATIO = 0.8;

    public MultithreadedTrainer(Network nn, TrainingDataset td) {
        super(nn, td);

        BatchedTrainingDataset btd = new BatchedTrainingDataset(td, nn.batch_size());
        BatchedTrainingDataset[] split = TrainTestSplit.train_test_split(btd, DEFAULT_TRAINING_RATIO);

        this.train_set = split[0];
        this.test_set = split[1];
        
        this.agents = new ThreadedAgent[train_set.batch_num()];
        this.test_agents = new ThreadedAgent[test_set.batch_num()];
        this.threads = new Thread[train_set.batch_num()];

        availableThreads = new RollingQueue<>(nn.max_threads());
        ready = new RollingQueue<>(nn.max_threads());
        prepareThreads();
    }

    public MultithreadedTrainer(Network nn, TrainingDataset td, double training_ratio) {
        super(nn, td);
        BatchedTrainingDataset btd = new BatchedTrainingDataset(td, nn.batch_size());
        BatchedTrainingDataset[] split = TrainTestSplit.train_test_split(btd, training_ratio);

        this.train_set = split[0];
        this.test_set = split[1];

        this.agents = new ThreadedAgent[train_set.batch_num()];
        this.test_agents = new ThreadedAgent[test_set.batch_num()];
        this.threads = new Thread[train_set.batch_num()];

        availableThreads = new RollingQueue<>(nn.max_threads() * 2);
        ready = new RollingQueue<>(nn.max_threads());
        prepareThreads();
    }

    private void prepareThreads() {
        int index = 0;
        for(TrainingDataset td: train_set) {
            ThreadedAgent ta = new ThreadedAgent(nn, this, td);
            agents[index++] = ta;
        }

        index = 0;
        if(test_set != null) {
            for(TrainingDataset td: test_set) {
                ThreadedAgent ta = new ThreadedAgent(nn, this, td);
                test_agents[index++] = ta;
            }
        }

        for(int i = 0; i < nn.max_threads(); i++) {
            availableThreads.pushHead(i);
        }
    }

    @Override
    public void train_batch() {
        nn.setDropoutEnable(true);

        if(train_set.batch_num() == 0) return;
        try {
            agents[index].setTrain(availableThreads.pop());
            threads[index] = new Thread(agents[index]);
            threads[index].start();

            threads[index].join();

            agents[index].setAdjust(availableThreads.pop());
            threads[index] = new Thread(agents[index]);
            threads[index].start();

            threads[index].join();

            index = (index + 1) % agents.length;
        } catch (InterruptedException e) {
            System.err.println(e.getMessage());
        }
    }

    @Override
    public void train_epoch() {
        nn.setDropoutEnable(true);


        if(train_set.batch_num() == 0) return;
        int trainIndex = 0;
        int adjustIndex = 0;
        while(trainIndex < agents.length || adjustIndex < agents.length) {

            // Predict phase

            while(availableThreads.size() > 0 && trainIndex < agents.length) {
                agents[trainIndex].setTrain(availableThreads.pop());
                threads[trainIndex] = new Thread(agents[trainIndex]);
                ready.push(threads[trainIndex]);
                trainIndex++;
            }

            while(ready.size() > 0) {
                ready.pop().start();
            }
            
            for(Thread t: threads) {
                if(t == null) continue;
                try {
                    t.join();
                } catch (InterruptedException e) {
                    System.err.println(e.getMessage());
                }
            }


            // Adjust phase

            while(availableThreads.size() > 0 && adjustIndex < trainIndex) {
                agents[adjustIndex].setAdjust(availableThreads.pop());
                threads[adjustIndex] = new Thread(agents[adjustIndex]);
                ready.push(threads[adjustIndex]);
                adjustIndex++;
            }

            while(ready.size() > 0) {
                ready.pop().start();
            }
            
            for(Thread t: threads) {
                if(t == null) continue;
                try {
                    t.join();
                } catch (InterruptedException e) {
                    System.err.println(e.getMessage());
                }
            }
        }
    }

    @Override
    public double computeTrainingEval() {
        nn.setDropoutEnable(false);

        double eval = 0;
        for(ThreadedAgent ta: agents) {
            eval += ta.computeTrainingEval();
        }
        return eval / train_set.batch_num();
    }

    @Override
    public double computeValidationEval() {
        nn.setDropoutEnable(false);
        
        double eval = 0;
        for(ThreadedAgent ta: test_agents) {
            eval += ta.computeValidationEval();
        }

        return eval / test_set.batch_num();
    }

    protected synchronized void finish(int threadIndex) {
        availableThreads.pushHead(threadIndex);
    }
    
}
