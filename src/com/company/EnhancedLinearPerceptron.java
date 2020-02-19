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

import java.text.DecimalFormat;

public class EnhancedLinearPerceptron implements Classifier, CapabilitiesHandler {

    private boolean debug = true;
    DecimalFormat df = new DecimalFormat("#.00");
    private double w[];

    // Additional Functionality flags for enhanced version
    private boolean standardisedAttributes = true;
    private boolean online = true;
    private boolean modelSelection = false;

    public void setStandardisedAttributes(boolean standardize) { this.standardisedAttributes = standardize; }
    public void setOnline(boolean online) {
        this.online = online;
    }
    public void setModelSelection(boolean modelSelection) { this.modelSelection = modelSelection; }

    private boolean selectModel(Instances instances) throws Exception{
        EnhancedLinearPerceptron ehp = new EnhancedLinearPerceptron();
        double cv_error_online = WekaTools.crossValError(ehp,instances);
        ehp.setOnline(false);
        double cv_error_offline = WekaTools.crossValError(ehp,instances);

        return cv_error_online <= cv_error_offline;
    }

    private void trainPerceptron(Instances instances, boolean online) throws Exception{
        w = new double[instances.numAttributes()]; // w (weights), array of weights for each attribute

        for(int i=0; i< w.length; i++){
            w[i] = 1; //(int)(Math.random()*10);
        }
        double n = 1; // n (learning rate) = 1
        double y; // predicted output y
        double t; // actual output

        int max_iterations = 20;
        int num_iterations = 0;
        do{
            double w_delta[] = new double[instances.numAttributes()];
            for(Instance instance: instances){
                int instanceIndex = num_iterations%instances.numInstances();
                num_iterations++; // Increase iteration count

                y = classifyInstance(instance); // Classify the instance
                y = y<0 ? -1 : 1; // Apply Logistic function to map y to -1 (if negative) or 1 (if y >= 0)

                t= instance.classValue(); // Get actual class value

                if(debug) System.out.print(num_iterations+"("+instanceIndex+")    y="+y+" t="+t);
                // If incorrect classification was made
                if(y!=t) {
                    if(debug) System.out.print("  Updating weights... "+ df.format(w[0]) + "," + df.format(w[1]));

                    // Update weights across all attributes
                    for (int i = 0; i < instances.numAttributes(); i++) {
                        // Calculate new weigh
                        w_delta[i] = 0.5 * n * (t - y) * instance.value(i); // Could ignore class value, but has no effect on classification
                        // If online, update weights immediately
                        if(online) w[i] = w[i] + w_delta[i];
                    }
                    if(debug) System.out.print("  --> "+ df.format(w[0]) + "," + df.format(w[1]));
                }
                if(debug) System.out.println();

                // If offline and just looked at last datapoint, update all weights
                if(!online && instanceIndex == instances.numInstances()-1 ){
                    // For every attribute/weight
                    for (int i = 0; i < instances.numAttributes(); i++) {
                        // Update weight with delta
                        w[i] = w[i] + w_delta[i];
                    }
                    w_delta = new double[instances.numAttributes()];
                }
            }

        }while(num_iterations < max_iterations); //Stopping function
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        if(this.standardisedAttributes){
            instances = WekaTools.standardize(instances);
        }
        boolean online = this.selectModel(instances);
        this.trainPerceptron(instances, this.online);
    }


//    private boolean Stopping(int num_iterations, int max_iterations){
//        return true ? (num_iterations > max_iterations) : false;
//    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        /*
        The method classifyInstance should applies the model to the new instance then applies
        a sensible decision rule to the resulting linear prediction.
        */
        double prediction_real = 0;
        for(int i = 0; i < w.length; i++){
            prediction_real = prediction_real + (w[i] * instance.value(i));
        }
        return prediction_real >= 0 ? 1.0 : -1.0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception { // TODO
        return new double[0];
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        //result.enable(Capabilities.Capability.MISSING_VALUES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        //result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.setMinimumNumberInstances(0);
        return result;
    }


    public static void main(String[] args) {
        Instances testData = WekaTools.loadClassificationData("src\\test_data.arff");
        testData.setClassIndex(2);
        EnhancedLinearPerceptron lp = new EnhancedLinearPerceptron();
        lp.setStandardisedAttributes(true);
        try{
            lp.setModelSelection(true);
            lp.buildClassifier(testData);
            //System.out.println(testData);

            //System.out.println(WekaTools.accuracy(lp, testData));
        }catch (Exception e){
            e.printStackTrace();
        }
    }



}

