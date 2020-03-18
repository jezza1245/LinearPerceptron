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

    public void setStandardisedAttributes(boolean standardize) { this.standardisedAttributes = standardize; }
    public void setOnline(boolean online) { this.online = online; }
    public void setModelSelection(boolean modelSelection) { this.modelSelection = modelSelection; }

    private boolean selectModel(Instances instances) throws Exception{
        EnhancedLinearPerceptron ehp = new EnhancedLinearPerceptron();
        ehp.setOnline(true);
        double cv_error_online = WekaTools.crossValError(ehp,instances);

        ehp = new EnhancedLinearPerceptron();
        ehp.setOnline(false);
        double cv_error_offline = WekaTools.crossValError(ehp,instances);

        return cv_error_online <= cv_error_offline;
    }

    private void trainPerceptron(Instances instances, boolean online) throws Exception{

        double y; // predicted output y
        double t; // actual output

        int numIterations = 0;
        int totalInstances= instances.numInstances();
        int iterationsSinceUpdate = 0;


        boolean revolutionWithoutpdate = false;
        boolean atIterationLimit = false;

        do{
            double weights_deltas[] = new double[instances.numAttributes()];

            int instanceIndex = numIterations%totalInstances;
            Instance instance = instances.get(instanceIndex);

            y = classifyInstance(instance); // Classify the instance
            y = y<0 ? -1 : 1; // Apply Logistic function to map y to -1 (if negative) or 1 (if y >= 0)

            t = instance.classValue(); // Get actual class value

            // If incorrect classification was made
            if(y!=t) {
                // Update weights across all attributes
                for (int i = 0; i < instances.numAttributes(); i++) {
                    // Calculate new weigh
                    weights_deltas[i] = 0.5 * learningRate * (t - y) * instance.value(i); // Could ignore class value, but has no effect on classification
                    // If online, update weights immediately
                    if(online) weights[i] = weights[i] + weights_deltas[i];
                }

                if(online) iterationsSinceUpdate = 0;
            }

            // If offline and just looked at last datapoint, update all weights
            if(!online && (instanceIndex == totalInstances-1) ){

                if(weights_deltas.equals(new double[instances.numAttributes()])){
                    break;
                }else {

                    // For every attribute/weight
                    for (int i = 0; i < instances.numAttributes(); i++) {
                        // Update weight with delta
                        weights[i] = weights[i] + weights_deltas[i];
                    }

                    weights_deltas = new double[instances.numAttributes()];
                }
            }else if(online){
                revolutionWithoutpdate = iterationsSinceUpdate >= (totalInstances-1); // Set flag if a full revolution has been made
            }
            System.out.println();

            numIterations++; // Increase iteration count

            atIterationLimit = numIterations >= maxIterations; // Set flag if at iteration limit

        }while(!revolutionWithoutpdate && !atIterationLimit); //Stopping function

        if(atIterationLimit && debug) System.out.println("Iteration limit reached");
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        weights = new double[instances.numAttributes()]; // weights (weights), array of weights for each attribute

        for(int i=0; i< weights.length; i++){
            weights[i] = 1; //(int)(Math.random()*10);
        }

        if(this.standardisedAttributes){ instances = WekaTools.standardize(instances); } //Standardize attributes
        if(this.modelSelection) { online = this.selectModel(instances); }
        this.trainPerceptron(instances, online);
    }


    public static void main(String[] args) {
        Instances testData = WekaTools.loadClassificationData("src\\test_data.arff");
        testData.setClassIndex(2);
        EnhancedLinearPerceptron lp = new EnhancedLinearPerceptron();
        try{
            lp.setModelSelection(false);
            lp.setStandardisedAttributes(false);
            lp.online = false;
            lp.buildClassifier(testData);

        }catch (Exception e){
            e.printStackTrace();
        }
    }

}

