package com.company;
import weka.classifiers.Classifier;

/*
A standard on-line Linear Perceptron classifier implemening the Weka Classifier
interface (or implemening AbstractClassifier).
Works when all attributes are continuous
If this is not the case it will throw an exception.
 */

import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

import java.io.Serializable;
import java.text.DecimalFormat;

public class EnhancedLinearPerceptron extends LinearPerceptron implements Serializable {

    static final long serialVersionUID = 42L;

    // Additional Functionality flags for enhanced version
    private boolean standardisedAttributes = true;
    private boolean online = true;
    private boolean modelSelection = false;
    Standardize standardize = new Standardize();


    public void setStandardisedAttributes(boolean standardize) { this.standardisedAttributes = standardize; }
    public void setOnline(boolean online) { this.online = online; }
    public void setModelSelection(boolean modelSelection) { this.modelSelection = modelSelection; }

    private boolean selectModel(Instances instances) throws Exception{
        EnhancedLinearPerceptron ehp = new EnhancedLinearPerceptron();
        int folds = 10;
        ehp.setOnline(true);
        double cv_error_online = WekaTools.crossValError(ehp,instances,folds);

        ehp = new EnhancedLinearPerceptron();
        ehp.setOnline(false);
        double cv_error_offline = WekaTools.crossValError(ehp,instances,folds);

        return cv_error_online <= cv_error_offline;
    }

    private void trainPerceptron(Instances instances) throws Exception{

        double y; // predicted output y
        double t; // actual output

        int numIterations = 0;
        int totalInstances= instances.numInstances();
        int iterationsSinceUpdate = 0;


        boolean revolutionWithoutUpdate = false;
        boolean atIterationLimit = false;

        double weights_deltas[] = new double[instances.numAttributes()];

        do{

            int instanceIndex = numIterations%totalInstances;
            Instance instance = instances.get(instanceIndex);

            y = classifyInstance(instance); // Classify the instance
            y = y<0 ? -1 : 1; // Apply Logistic function to map y to -1 (if negative) or 1 (if y >= 0)

            t = instance.classValue(); // Get actual class value
            if(t==0) t=-1; //If class given is 0, set to -1 to work with perceptron logic

            // If incorrect classification was made
            if(y!=t) {
                // Update weights across all attributes
                for (int i = 0; i < instances.numAttributes(); i++) {
                    // Calculate new weigh
                    weights_deltas[i] = 0.5 * learningRate * (t - y) * instance.value(i); // Could ignore class value, but has no effect on classification
                    // If online, update weights immediately
                    if(online) weights[i] = weights[i] + weights_deltas[i];
                }

                iterationsSinceUpdate = 0;
            }else{
                iterationsSinceUpdate++;
            }

            // If offline and just looked at last datapoint, update all weights
            if(!online && (instanceIndex == totalInstances-1) ){
                // no corrections made

                boolean changes = false;
                for(double weight: weights_deltas){
                    if(weight!= 0.0) changes = true;
                }

                if(changes) {
                    // For every attribute/weight
                    for (int i = 0; i < instances.numAttributes(); i++) {
                        // Update weight with delta
                        weights[i] = weights[i] + weights_deltas[i];
                    }

                    weights_deltas = new double[instances.numAttributes()];
                }else{
                    break;
                }

            }

            numIterations++; // Increase iteration count
            atIterationLimit = numIterations >= maxIterations; // Set flag if at iteration limit
            revolutionWithoutUpdate = (online && ( iterationsSinceUpdate >= (totalInstances-1) ) ); // Set flag if a full revolution has been made


        }while(!revolutionWithoutUpdate && !atIterationLimit); //Stopping function

        if(revolutionWithoutUpdate && debug) System.out.println("Iterated through all instances without update.");
        if(atIterationLimit && debug) System.out.println("Iteration limit reached");
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        weights = new double[instances.numAttributes()]; // weights (weights), array of weights for each attribute

        if(randomizeStartingCondition){
            for(int i=0; i< weights.length; i++){
                weights[i] = (int)(Math.random()*10);
            }
        }
        else {
            for (int i = 0; i < weights.length; i++) {
                weights[i] = 1; //(int)(Math.random()*10);
            }
        }

        if(this.standardisedAttributes){
            standardize.setInputFormat(instances);
            instances = Filter.useFilter(instances, standardize);
        } //Standardize attributes

        if(this.modelSelection) { online = this.selectModel(instances); }
        this.trainPerceptron(instances);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {

        if(standardisedAttributes){
            standardize.input(instance);
            instance = standardize.output();
        }

        double prediction_real = 0;
        for(int i = 0; i < weights.length; i++){
            prediction_real = prediction_real + (weights[i] * instance.value(i));
        }

        return prediction_real >= 0 ? 1.0 : -1.0;
    }


    public static void main(String[] args) {
//        Instances testData = WekaTools.loadClassificationData("resources\\test_data.arff");
//        testData.setClassIndex(2);
//        EnhancedLinearPerceptron lp = new EnhancedLinearPerceptron();
//        try{
//            lp.setModelSelection(false);
//            lp.setStandardisedAttributes(false);
//            lp.online = false;
//            lp.buildClassifier(testData);
//            System.out.println(WekaTools.accuracy(lp,testData));
//
//        }catch (Exception e){
//            e.printStackTrace();
//        }

        try{

            Instances blood[] = WekaTools.getDataSetSplit("blood");
            Instances train = blood[0];
            Instances test = blood[1];

            train.setClassIndex(train.numAttributes()-1);
            test.setClassIndex(test.numAttributes()-1);

            EnhancedLinearPerceptron classifier = new EnhancedLinearPerceptron();
            classifier.online = false;
            classifier.standardisedAttributes = true;
            classifier.modelSelection = false;
            classifier.buildClassifier(train);

            System.out.println(WekaTools.accuracy(classifier,test));

        }catch(Exception e){
            e.printStackTrace();
        }
    }

}

