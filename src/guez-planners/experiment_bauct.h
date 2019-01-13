#pragma once

#include "bauct.h"
#include "simulator.h"
#include "statistic.h"
#include <fstream>
#include "utils.h"
#include "samplerFactory.h"

#include "experiment.h"
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------

class EXPERIMENT_bauct
{
public:

    struct PARAMS
    {
        PARAMS();
        
        int NumRuns;
        int NumSteps;
        int SimSteps;
        double TimeOut;
        double Accuracy;
        int UndiscountedHorizon;
        bool AutoExploration;
    };

    EXPERIMENT_bauct(const SIMULATOR& real, const SIMULATOR& simulator, 
        const std::string& outputFile, 
        EXPERIMENT_bauct::PARAMS& expParams, BAUCT::PARAMS& searchParams,
				SamplerFactory& _samplerFact);

    void Run(std::vector<double>& Rhist);
		void RunBandit(std::vector<uint>& Rhist, std::vector<uint>& optArm, uint bestArm);

private:

    const SIMULATOR& Real;
    const SIMULATOR& Simulator;
    EXPERIMENT_bauct::PARAMS& ExpParams;
    BAUCT::PARAMS& SearchParams;
    RESULTS Results;
		
		SamplerFactory& samplerFact;
    std::ofstream OutputFile;
};

//----------------------------------------------------------------------------

