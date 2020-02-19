package com.company;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RandomSubset;

import java.io.EOFException;
import java.util.Random;

public class LinearPerceptronEnsemble implements Classifier {

    int ensembleSize;
    LinearPerceptron ensemble[];
    double attributeProportion = 0.5;

    public LinearPerceptronEnsemble(){
        ensembleSize = 50;
        ensemble = new LinearPerceptron[ensembleSize];
        this.initializeEnsemble();
    }
    public LinearPerceptronEnsemble(int ensembleSize){
        this.ensembleSize = ensembleSize;
        ensemble = new LinearPerceptron[ensembleSize];
        this.initializeEnsemble();
    }

    public void setAttributePortion(double attributePortion){
        this.attributeProportion = attributePortion;
    }

    private void initializeEnsemble(){
        for(int i=0; i<this.ensembleSize; i++){
            ensemble[i] = new LinearPerceptron();
        }
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {

        RandomSubset rs = new RandomSubset();
        rs.setInputFormat(instances);
        rs.setNumAttributes(attributeProportion);

        int count = 0;
        for(LinearPerceptron classifier: ensemble){
            int random = count++;
            rs.setSeed(random);
            Instances newInstances = new Instances(Filter.useFilter(instances, rs));
            System.out.println(newInstances);
        }
        System.out.println("done");
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }




    public static void main(String[] args) {
        Instances testData = WekaTools.loadClassificationData("src\\test_data.arff");
        testData.setClassIndex(2);
        LinearPerceptronEnsemble lpe = new LinearPerceptronEnsemble();
        try{
            lpe.setAttributePortion(0.5);
            lpe.buildClassifier(testData);
        }catch (Exception e){
            e.printStackTrace();
        }
        System.out.println("hello");
    }
}
