#pragma once
#include <rapidjson/document.h>
#include <ostream>
#include <string>
#include "Options.cuh"

struct ExperimentPhase {
    std::string phaseName;
    Options opts;
};

struct ExperimentParameters {
    std::string experimentName;
    ExperimentPhase phases[5];
};

class ExperimentReader {
   public:
    const std::string jsonFname;
    const Options defaults;
    rapidjson::Document doc;
    ExperimentParameters experiment;
    ExperimentReader(const std::string& jsonFname, Options defaults);
    std::string readFile(const std::string& jsonFname);
    void parseJson();
    ExperimentPhase parsePhase(rapidjson::Value& phaseJson);
    void parseOptions(rapidjson::Value& optionsJson, Options& opts);
    void parseInt(rapidjson::Value& element, const std::string& key,
                  int& out_target, bool mandatory = false);
    void parseFloat(rapidjson::Value& element, const std::string& key,
                    float& out_target, bool mandatory = false);
    void parseBool(rapidjson::Value& element, const std::string& key,
                   bool& out_target, bool mandatory = false);
    void asserts(bool assertion, std::string text);
};

std::ostream& operator<<(std::ostream& os, const ExperimentReader& or);