package org.deeplearning4j.examples.pattern_recognition_final_RBM;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.stepfunctions.NegativeDefaultStepFunction;
import org.deeplearning4j.optimize.stepfunctions.StepFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.Random;

/**
 * Created by agibsonccc on 9/11/14.
 *
 * ***** NOTE: This example has not been tuned. It requires additional work to produce sensible results *****
 */
public class Pattern_Final_RBM {

    private static Logger log = LoggerFactory.getLogger(Pattern_Final_RBM.class);

    public static void main(String[] args) throws Exception {
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int numSamples = 60000; //60000;
        int batchSize = 1000; //100;
        int iterations = 20; //10;
        int seed = 123;
        int listenerFreq = 1/5; //batchSize / 5;

        double rate = 0.0015;
        log.info("Load data....");
        DataSetIterator iter = new MnistDataSetIterator(batchSize, numSamples, true, true, true, seed);

        DataSet next = iter.next();
        //DataSet next = new MnistDataFetcher();
        next.shuffle();
        next.normalizeZeroMeanZeroUnitVariance();

        log.info("Split data...");
        int splitTrainNum = (int) (batchSize * .8);
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum, new Random(seed));
        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();

        //Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()

            //.stepFunction(stepfunc)

            .seed(seed)

            //.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            //.gradientNormalizationThreshold(1.0)

            .weightInit(WeightInit.XAVIER)
            .iterations(iterations)
            .learningRate(rate) // Added by myself

            //.updater(Updater.NESTEROVS).momentum(0.98)
            //.regularization(true).l2(rate * 0.005)
            .l1(1e-1).regularization(true).l2(2e-4)

            //.useDropConnect(true)
            //.momentum(0.5)
            //.momentumAfter(Collections.singletonMap(3, 0.9))

            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)//CONJUGATE_GRADIENT)
            .list(4)
            .layer(0, new RBM.Builder()//RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                .activation("relu")
                .nIn(numRows*numColumns).nOut(700)
                .lossFunction(LossFunction.RMSE_XENT)
                .build())
            .layer(1, new RBM.Builder()//RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                .activation("relu")
                .nIn(700).nOut(500)
                .lossFunction(LossFunction.RMSE_XENT)
                .build())
            .layer(2, new RBM.Builder()//RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                .activation("relu")
                .nIn(500).nOut(250)
                .lossFunction(LossFunction.RMSE_XENT)
                .build())
            .layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation("softmax")
                .nIn(250).nOut(outputNum)
                .build())
            .pretrain(true)
            .backprop(true)
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train model....");
        model.fit(train); // achieves end to end pre-training

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);

        //DataSetIterator testIter = new MnistDataSetIterator(100,10000);
        //DataSetIterator testIter = new MnistDataSetIterator(batchSize, false, seed);


//        while(testIter.hasNext()) {
//            DataSet testMnist = next1.next();
//            INDArray predict21 = model.output(testMnist.getFeatureMatrix());
//            eval.eval(testMnist.getLabels(), predict21);
//        }

        eval.eval(test.getLabels(), model.output(test.getFeatureMatrix(), Layer.TrainingMode.TEST));

        log.info(eval.stats());
        log.info("****************Example finished********************");

    }

}
