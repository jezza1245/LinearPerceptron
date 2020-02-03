package com.company;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.util.Random;

public class WekaTools {

    public static Instances loadClassificationData(String filePath){
        Instances data;
        try{
            FileReader reader = new FileReader(filePath);
            data = new Instances(reader);
        }catch(Exception e){
            System.out.println("Exception caught: "+e);
            data = null;
        }
        return data;
    }

    public static Instances[] splitData(Instances all, double proportion){

        Instances split[] = new Instances[2];
        int totalInstances = all.numInstances();

        int splitAt = (int) (proportion*totalInstances);

        all.randomize(new Random(42));
        split[0] = new Instances(all, 0, splitAt);
        split[1] = new Instances(all, splitAt, totalInstances-splitAt);


        return split;
    }

    public static double[] classDistributionsAcrossInstances(Instances instances) throws weka.core.UnassignedClassException{


        try{

            double distributions[] = new double[instances.numClasses()];

            for(Instance instance: instances){
                double cls = instance.classValue();
                distributions[(int)cls]++;
            }

            return distributions;
        }catch (weka.core.UnassignedClassException e){
            throw e;
        }

    }


    public static double accuracy(Classifier classifier, Instances test) {
        int totalInstances = test.numInstances();
        int numCorrect = 0;

        Attribute classAttribute = test.classAttribute();
        for (Instance instance : test) {
            try {
                double predicted = classifier.classifyInstance(instance);
                double actual = instance.value(classAttribute);
                if (predicted == actual) {
                    numCorrect++;
                }
            } catch (Exception e) {
                System.out.println("Error classifying instance");

            }
        }

        double acc = numCorrect/(double)totalInstances;

        return acc;
    }

    public static void main(String[] args) {

    }


}
