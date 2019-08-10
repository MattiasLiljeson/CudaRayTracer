#include "ExperimentReader.h"
#include <fstream>
#include <vector>
#include "Mat.cuh"
#include "Options.cuh"
#include "Popup.h"

ExperimentReader::ExperimentReader(const std::string& jsonFname,
                                   Options defaults)
    : jsonFname(jsonFname), defaults(defaults) {
    std::string json = readFile(jsonFname);
    doc.Parse(json.c_str());
    parseJson();
}

std::string ExperimentReader::readFile(const std::string& jsonFname) {
    std::string json;
    std::ifstream myfile;
    myfile.open(jsonFname);
    if (myfile.is_open()) {
        std::string line;
        while (getline(myfile, line)) {
            json += line += '\n';
        }
    }
    myfile.close();
    return json;
}

void ExperimentReader::parseJson() {
    ExperimentParameters exp;
    // exp.opts = defaults;
    asserts(doc["name"].IsString(),
            "The experiment must have a name, of type string!");
    exp.experimentName = doc["name"].GetString();
    asserts(doc["phases"].IsArray(),
            "The experiment must have a list of phases!");
    // asserts(doc["phases"].Size()==5,
    //        "The experiment phase list must be of length 5!");
    rapidjson::Value& phases = doc["phases"];
    for (rapidjson::SizeType i = 0; i < phases.Size(); ++i) {
        exp.phases[i] = parsePhase(phases[i]);
    }
    experiment = exp;
}

ExperimentPhase ExperimentReader::parsePhase(rapidjson::Value& phaseJson) {
    ExperimentPhase phase;
    asserts(phaseJson["name"].IsString(),
            "The phase must have a name, of type string!");
    phase.phaseName = phaseJson["name"].GetString();

    phase.opts = defaults;
    parseOptions(phaseJson["options"], phase.opts);
    return phase;
}

void ExperimentReader::parseOptions(rapidjson::Value& optionsJson,
                                    Options& opts) {
    if (optionsJson.HasMember("model") && optionsJson["model"].IsObject()) {
        rapidjson::Value& model = optionsJson["model"];
        opts.host.model.folder = model["folder"].GetString();
        opts.host.model.fname = model["fname"].GetString();

        if (model.HasMember("transformationMatrix") &&
            model["transformationMatrix"].IsArray()) {
            rapidjson::Value& transformMat = model["transformationMatrix"];
            asserts(model["translate"].Size() != 3,
                    "The translate model.transformationMatrix must be of "
                    "length 16!");
            for (rapidjson::SizeType i = 0; i < transformMat.Size(); i++) {
                opts.host.model.transform[i][0] = transformMat[i].GetFloat();
                opts.host.model.transform[i][1] = transformMat[i].GetFloat();
                opts.host.model.transform[i][2] = transformMat[i].GetFloat();
                opts.host.model.transform[i][3] = transformMat[i].GetFloat();
            }
        }
        std::vector<Mat44f> transformations;
        if (model.HasMember("scale")) {
            asserts(model["scale"].IsFloat(),
                    "The translate model.member must be a float!");
            transformations.push_back(Mat44f::scale(model["scale"].GetFloat()));
        }
        if (model.HasMember("translate")) {
            asserts(model["translate"].IsArray(),
                    "The translate model.member must be an array!");
            asserts(model["translate"].Size() == 3,
                    "The translate model->translate must be of length 3!");
            transformations.push_back(
                Mat44f::translate(model["translate"][0].GetFloat(),
                                  model["translate"][1].GetFloat(),
                                  model["translate"][2].GetFloat()));
        }
        if (!transformations.empty()) {
            Mat44f finalMat = Mat44f::identity();
            for (auto t : transformations) {
                finalMat = finalMat * t;
            }
            opts.host.model.transform = finalMat;
        }
    }
    parseInt(optionsJson, "lightCnt", opts.host.lightCnt);
    parseFloat(optionsJson, "fov", opts.host.fov);
    parseInt(optionsJson, "width", opts.device.width);
    parseInt(optionsJson, "height", opts.device.height);
    parseFloat(optionsJson, "shadowBias", opts.device.shadowBias);
    parseInt(optionsJson, "blockSize", opts.device.blockSize);
    parseInt(optionsJson, "maxDepth", opts.device.maxDepth);
    parseInt(optionsJson, "samples", opts.device.samples);
    parseBool(optionsJson, "gammaCorrection", opts.device.gammaCorrection);
}

void ExperimentReader::parseInt(rapidjson::Value& element,
                                const std::string& key, int& out_target,
                                bool mandatory) {
    if (mandatory) {
        asserts(element.HasMember(key.c_str()), "'" + key + "' must exist!");
    }
    if (element.HasMember(key.c_str())) {
        asserts(element[key.c_str()].IsInt(),
                "'" + key + "' must be of type int!");
        out_target = element[key.c_str()].GetInt();
    }
}

void ExperimentReader::parseFloat(rapidjson::Value& element,
                                  const std::string& key, float& out_target,
                                  bool mandatory) {
    if (mandatory) {
        asserts(element.HasMember(key.c_str()), "'" + key + "' must exist!");
    }
    if (element.HasMember(key.c_str())) {
        asserts(element[key.c_str()].IsFloat(),
                "'" + key + "' must be of type float!");
        out_target = element[key.c_str()].GetFloat();
    }
}

void ExperimentReader::parseBool(rapidjson::Value& element,
                                 const std::string& key, bool& out_target,
                                 bool mandatory) {
    if (mandatory) {
        asserts(element.HasMember(key.c_str()), "'" + key + "' must exist!");
    }
    if (element.HasMember(key.c_str())) {
        asserts(element[key.c_str()].IsBool(),
                "'" + key + "' must be of type bool!");
        out_target = element[key.c_str()].GetBool();
    }
}

void ExperimentReader::asserts(bool assertion, std::string text) {
    if (!assertion) {
        Popup::error("Failed to parse json file: '" + jsonFname + "'.\n" +
                     text);
        // exit(1); //todo: throw instead?
    }
}

std::ostream& operator<<(std::ostream& os, const ExperimentReader& or) {
    os << "file: " << or.jsonFname << std::endl;
    os << "exp name: " << or.experiment.experimentName << std::endl;
    for (int i = 0; i < 5; ++i) {
        os << "\tphase name: " << or.experiment.phases[i].phaseName
                                        << std::endl;
        os << "\t\tphase model: "
           << or.experiment.phases[i].opts.host.model.fname << std::endl;
    };
    return os;
}