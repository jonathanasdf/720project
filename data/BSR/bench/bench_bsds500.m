addpath benchmarks

clear all;close all;clc;

imgDir = '../images/test';
gtDir = '../groundTruth/test';
inDir = '../detected';
outDir = '../detected_eval';
mkdir(outDir);

tic;
boundaryBench(imgDir, gtDir, inDir, outDir)
toc;

plot_eval(outDir);