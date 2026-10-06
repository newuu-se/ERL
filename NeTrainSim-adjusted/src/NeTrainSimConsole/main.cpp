#include <QCoreApplication>
#include <QLocale>
#include <QTranslator>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include "traindefinition/train.h"
#include "traindefinition/trainslist.h"
#include "network/network.h"
#include "simulator.h"
#include "util/vector.h"
#include <iostream>
#include <sstream>
#include <QCoreApplication>
#include <QCommandLineParser>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <stdio.h>
#include <filesystem>
#include <algorithm>
#include "qdir.h"
#include "simulatorapi.h"
#include "errorhandler.h"
#include "VersionConfig.h"

const std::string compilation_date = __DATE__;
const std::string compilation_time = __TIME__;
const QString NETWORK_NAME = "mainNetwork";
/**
 * @brief checkParserValue
 * @param parser
 * @param option
 * @return
 */
bool checkParserValue(QCommandLineParser& parser, const QCommandLineOption &option, std::string s, bool isRequired = true){
    if(parser.isSet(option)) {
        return true;
    }
    if (isRequired){
        fputs(qPrintable(QString::fromStdString(s)), stdout);
        fputs("\n\n", stdout);
        fputs(qPrintable(parser.helpText()), stdout);
    }
    return false;
}

/**
 * Main entry-point for this application
 *
 * @author	Ahmed Aredah
 * @date	2/28/2023
 *
 * @param 	argc	The number of command-line arguments provided.
 * @param 	argv	An array of command-line argument strings.
 *
 * @returns	Exit-code for the process - 0 for success, else an error code.
 */
int main(int argc, char *argv[])
{
    // ####################################################
    // #                     values                       #
    // ####################################################
    std::string GithubLink = "https://github.com/VTTI-CSM/NeTrainSim";

    QCoreApplication app(argc, argv);
    QTranslator translator;
    const QStringList uiLanguages = QLocale::system().uiLanguages();
    for (const QString &locale : uiLanguages) {
        const QString baseName = "NeTrainSim_" + QLocale(locale).name();
        if (translator.load(":/i18n/" + baseName)) {
            app.installTranslator(&translator);
            break;
        }
    }
    QCoreApplication::setApplicationName(NeTrainSim_NAME);
    QCoreApplication::setApplicationVersion(NeTrainSim_VERSION);
    QCoreApplication::setOrganizationName(NeTrainSim_VENDOR);

    QCommandLineParser parser;
    parser.setApplicationDescription("Open-source network train simulator");
    QCommandLineOption helpOption(QStringList() << "h" << "help" << "?",
                                  "Display this help message.");
    parser.addOption(helpOption);
    //parser.addHelpOption();
    parser.addVersionOption();


    const QCommandLineOption nodesOption(QStringList() << "n" << "nodes",
                                         QCoreApplication::translate("main", "[Required] the nodes filename."), "nodesFile", "");
    parser.addOption(nodesOption);

    const QCommandLineOption linksOption(QStringList() << "l" << "links",
                                         QCoreApplication::translate("main", "[Required] the links filename."), "linksFile", "");
    parser.addOption(linksOption);

    const QCommandLineOption trainsOption(QStringList() << "t" << "trains",
                                          QCoreApplication::translate("main", "[Required] the trains filename."), "trainsFile", "");
    parser.addOption(trainsOption);

    const QCommandLineOption outputLocationOption(QStringList() << "o" << "output",
                                                  QCoreApplication::translate("main", "[Optional] the output folder address. \nDefault is 'C:\\Users\\<USERNAME>\\Documents\\NeTrainSim\\'."), "outputLocation", "");
    parser.addOption(outputLocationOption);

    const QCommandLineOption summaryFilenameOption(QStringList() << "s" << "summary",
                                                   QCoreApplication::translate("main", "[Optional] the summary filename. \nDefault is 'trainSummary_timeStamp.txt'."), "summaryFilename", "");
    parser.addOption(summaryFilenameOption);

    const QCommandLineOption summaryExportAllOption(QStringList() << "a" << "all",
                                                    QCoreApplication::translate("main", "[Optional] bool to show summary of all trains in the summary file. \nDefault is 'false'."), "summarizeAllTrains", "false");
    parser.addOption(summaryExportAllOption);

    const QCommandLineOption exportInstaTrajOption(QStringList() << "e" << "export",
                                                   QCoreApplication::translate("main", "[Optional] bool to export instantaneous trajectory. \nDefault is 'false'."), "exportTrajectoryOptions" ,"false");
    parser.addOption(exportInstaTrajOption);

    const QCommandLineOption instaTrajOption(QStringList() << "i" << "insta",
                                             QCoreApplication::translate("main", "[Optional] the instantaneous trajectory filename. \nDefault is 'trainTrajectory_timeStamp.csv'."), "instaTrajectoryFile", "");
    parser.addOption(instaTrajOption);

    const QCommandLineOption timeStepOption(QStringList() << "p" << "timeStep",
                                            QCoreApplication::translate("main", "[Optional] the simulator time step. \nDefault is '1.0'."), "simulatorTimeStep", "1.0");
    parser.addOption(timeStepOption);

    const QCommandLineOption enableOptimization(QStringList() << "z" << "optimization",
                                                QCoreApplication::translate("main", "[Optional] bool to enable single train trajectory optimization. \nDefault is 'false'"), "optimization", "false");
    parser.addOption(enableOptimization);

    const QCommandLineOption optimizationEvery(QStringList() << "y" << "optimizeEvery",
                                               QCoreApplication::translate("main", "[Optional] the optimization frequency. \nDefault is '1'."), "optimizeEvery", "1");
    parser.addOption(optimizationEvery);

    const QCommandLineOption optimizationLookahead(QStringList() << "d" << "optimizerLookahead",
                                                   QCoreApplication::translate("main", "[Optional] the forward lookahead distance for the optimizer. \nDefault is '1'."), "optimizerLookahead", "1");
    parser.addOption(optimizationLookahead);

    const QCommandLineOption optimizationSpeedPriorityFactor(QStringList() << "k" << "OptimizationSpeedFactor",
                                                             QCoreApplication::translate("main", "[Optional] the speed priority factor in case of optimization. \n Default is '0.0'."), "OptimizationSpeedFactor", "0.0");
    parser.addOption(optimizationSpeedPriorityFactor);

    const QCommandLineOption interactiveOption(QStringList() << "I" << "interactive",
                                               QCoreApplication::translate("main", "[Optional] enable interactive RL mode (JSON over stdin/stdout)."));
    parser.addOption(interactiveOption);

    // process all the arguments
    parser.process(app);

    // display the help if requested and exit
    if (parser.isSet(helpOption)) {
        parser.showHelp(0);
    }

    // show app details
    std::stringstream hellos;
    hellos << NeTrainSim_NAME << " [Version " << NeTrainSim_VERSION << ", "  << compilation_date << " " << compilation_time << " Build" <<  "]" << endl;
    hellos << NeTrainSim_VENDOR << endl;
    hellos << GithubLink << endl;
    std::cout << hellos.str() << "\n";




    std::string nodesFile, linksFile, trainsFile, exportLocation, summaryFilename, instaTrajFilename;
    bool exportInstaTraj = false;
    double timeStep = 1.0;
    bool optimize = false;
    double optimize_speedfactor = 0.0;
    int optimizerFrequency = 0;
    int lookahead = 0;
    bool interactiveMode = false;

    // read values from the cmd
    // read required values

    if (checkParserValue(parser, nodesOption, "nodes file is missing!", true)) { nodesFile = parser.value(nodesOption).toStdString(); }
    else { return 1;}
    if (checkParserValue(parser, linksOption, "links file is missing!", true)) { linksFile = parser.value(linksOption).toStdString(); }
    else { return 1;}
    if (checkParserValue(parser, trainsOption, "trains file is missing!", true)) { trainsFile = parser.value(trainsOption).toStdString(); }
    else { return 1;}

    // read optional values
    if (checkParserValue(parser, outputLocationOption, "" ,false)){
        exportLocation = parser.value(outputLocationOption).toStdString();
        QDir directory(QString::fromStdString(exportLocation));
        // check if directory is valid
        if (!directory.exists()) {
            fputs(qPrintable("export directory is not valid!"), stdout);
            return 1;
        }
    }
    else { exportLocation = ""; }

    if (checkParserValue(parser, summaryFilenameOption, "", false)){ summaryFilename = parser.value(summaryFilenameOption).toStdString(); }
    else { summaryFilename = ""; }

    if (checkParserValue(parser, exportInstaTrajOption, "", false)){
        stringstream ss(parser.value(exportInstaTrajOption).toStdString());
        ss >> std::boolalpha >> exportInstaTraj;
    }
    else { exportInstaTraj = false; }

    if (checkParserValue(parser, instaTrajOption, "", false)){ instaTrajFilename = parser.value(instaTrajOption).toStdString(); }
    else { instaTrajFilename = ""; }

    if (checkParserValue(parser, timeStepOption, "", false)) { timeStep = parser.value(timeStepOption).toDouble(); }
    else { timeStep = 1.0; }

    if (checkParserValue(parser, enableOptimization, "", false)){
        stringstream ss(parser.value(enableOptimization).toStdString());
        ss >> std::boolalpha >> optimize;
    }
    else { optimize = false; }

    if (checkParserValue(parser, optimizationEvery, "", false)) { optimizerFrequency = parser.value(optimizationEvery).toInt(); }
    else { optimizerFrequency = 1.0; }
    if (checkParserValue(parser, optimizationLookahead, "", false)) { lookahead = parser.value(optimizationLookahead).toInt(); }
    else { lookahead = 1.0; }

    if (checkParserValue(parser, optimizationSpeedPriorityFactor, "", 0.0)) {optimize_speedfactor = parser.value(optimizationSpeedPriorityFactor).toDouble(); }
    else { optimize_speedfactor = 0.0;}

    interactiveMode = parser.isSet(interactiveOption);

    try {
        std::cout << "Reading Trains!                 \r";

        SimulatorAPI::ContinuousMode::createNewSimulationEnvironmentFromFiles(
            QString::fromStdString(nodesFile),
            QString::fromStdString(linksFile),
            NETWORK_NAME, QString::fromStdString(trainsFile),
            timeStep);

        auto trains = SimulatorAPI::ContinuousMode::getTrains(NETWORK_NAME);

        for (auto &t: trains) {
            QEventLoop::connect(t.get(), &Train::slowSpeedOrStopped,
                    [](const auto &msg){
                        ErrorHandler::showWarning(msg);});

            QEventLoop::connect(t.get(), &Train::suddenAccelerationOccurred,
                    [](const auto &msg){
                ErrorHandler::showWarning(msg);});

            t->setOptimization(optimize, optimize_speedfactor,
                               optimizerFrequency, lookahead);
        }
        Simulator* sim =
            SimulatorAPI::ContinuousMode::getSimulator(NETWORK_NAME);

        if (exportLocation != "" ) { sim->setOutputFolderLocation(exportLocation); }
        if (summaryFilename != "" ) { sim->setSummaryFilename(summaryFilename); }

        sim->setExportInstantaneousTrajectory(exportInstaTraj,
                                              instaTrajFilename);

        if (interactiveMode) {
            sim->initializeSimulator(false);
            std::cout << "Starting the Simulator in interactive mode.\n";
            std::cout.flush();

            auto emitInteractiveState = [&]() {
                QJsonObject state;
                auto currentTrains = SimulatorAPI::ContinuousMode::getTrains(NETWORK_NAME);
                const bool terminated = sim->checkAllTrainsReachedDestination();
                const double timestep = sim->getCurrentStateAsJson()["CurrentSimulationTime"].toDouble();

                state["timestep"] = timestep;
                state["speed_mps"] = 0.0;
                state["position_m"] = 0.0;
                state["grade_perc"] = 0.0;
                state["curvature_perc"] = 0.0;
                state["remaining_dist_m"] = 0.0;
                state["energy_kwh"] = 0.0;
                state["link_max_speed_mps"] = 0.0;
                state["terminated"] = terminated;
                state["notch"] = 0;

                if (!currentTrains.empty()) {
                    const auto &train = currentTrains.front();
                    double grade = 0.0;
                    double curvature = 0.0;
                    double linkMaxSpeed = 0.0;
                    int notch = 0;

                    // Direction-aware grade: the link's grade map holds
                    // {fromNode: +g, toNode: -g}; the physics indexes it with the
                    // node the tip last passed (previousNodeID). Using begin()
                    // (lowest node id) is only correct for ascending-id travel —
                    // on a return trip (e.g. path 1500→1) it inverts the sign.
                    const auto directionalGrade =
                        [&train](const std::map<int, double> &gmap) -> double {
                        if (gmap.empty()) { return 0.0; }
                        const auto it = gmap.find(train->previousNodeID);
                        return (it != gmap.end()) ? it->second
                                                  : gmap.begin()->second;
                    };

                    if (train->currentFirstLink != nullptr) {
                        grade = directionalGrade(train->currentFirstLink->grade);
                        curvature = train->currentFirstLink->curvature;
                        linkMaxSpeed = train->currentFirstLink->freeFlowSpeed;
                    } else if (!train->currentLinks.empty()) {
                        auto fallbackLink = train->currentLinks.front();
                        if (fallbackLink != nullptr) {
                            grade = directionalGrade(fallbackLink->grade);
                            curvature = fallbackLink->curvature;
                            linkMaxSpeed = fallbackLink->freeFlowSpeed;
                        }
                    }

                    if (!train->locomotives.empty()) {
                        notch = train->locomotives.front()->currentLocNotch;
                    }

                    state["speed_mps"] = train->currentSpeed;
                    state["position_m"] = train->travelledDistance;
                    state["grade_perc"] = grade;
                    state["curvature_perc"] = curvature;
                    state["remaining_dist_m"] = std::max(0.0, train->trainTotalPathLength - train->travelledDistance);
                    state["energy_kwh"] = train->energyStat;
                    state["link_max_speed_mps"] = linkMaxSpeed;
                    state["terminated"] = train->reachedDestination;
                    state["notch"] = notch;
                }

                const QByteArray json = QJsonDocument(state).toJson(QJsonDocument::Compact);
                std::cout << "NTS_JSON " << json.constData() << std::endl;
                std::cout.flush();
            };

            std::string inputLine;
            while (std::getline(std::cin, inputLine)) {
                // Parse action JSON; finalize before re-throwing so output files flush.
                int notch = 0;
                try {
                    QJsonParseError parseError;
                    const QJsonDocument inputDoc = QJsonDocument::fromJson(
                        QByteArray::fromStdString(inputLine), &parseError);
                    if (parseError.error != QJsonParseError::NoError || !inputDoc.isObject()) {
                        throw std::runtime_error("Interactive input must be a JSON object, e.g. {\"notch\": 3}");
                    }
                    notch = inputDoc.object().value("notch").toInt(0);
                    notch = std::clamp(notch, 0, 8);
                } catch (...) {
                    sim->finalizeSimulation();
                    QCoreApplication::processEvents();
                    throw;
                }

                // Apply RL notch directly via override path — bypasses the built-in
                // A* optimizer so the agent's action is the actual throttle used.
                auto currentTrains = SimulatorAPI::ContinuousMode::getTrains(NETWORK_NAME);
                for (auto &train : currentTrains) {
                    train->optimize = false;

                    int maxNotch = 8;
                    if (!train->locomotives.empty()) {
                        maxNotch = std::max(1, train->locomotives.front()->Nmax);
                    }
                    const int appliedNotch = std::clamp(notch, 0, maxNotch);
                    const double throttle = std::pow(
                        static_cast<double>(appliedNotch) / static_cast<double>(maxNotch), 2.0);

                    for (auto &locomotive : train->locomotives) {
                        locomotive->rlOverrideEnabled = true;
                        locomotive->rlOverrideThrottle = throttle;
                        locomotive->currentLocNotch = appliedNotch;
                    }
                }

                sim->runOneTimeStep();
                emitInteractiveState();
                // Drain queued Qt events to prevent use-after-free in SimulatorAPI
                // singleton (same issue documented after runSimulation below).
                QCoreApplication::processEvents();

                if (sim->checkAllTrainsReachedDestination()) {
                    break;
                }
            }

            sim->finalizeSimulation();
            QCoreApplication::processEvents();
            std::cout << "Output folder: " << sim->getOutputFolder() << std::endl;
        } else {
            // run the actual simulation
            std::cout <<"Starting the Simulator!                                "
                         "              \n";
            sim->runSimulation();
            // Drain any queued Qt events posted during simulation (e.g. from
            // queued signal connections in SimulatorAPI) before QCoreApplication
            // is destroyed.  Without this the static SimulatorAPI singleton
            // outlives QCoreApplication and the deferred event delivery triggers
            // a use-after-free segfault during app destruction.
            QCoreApplication::processEvents();
            std::cout << "Output folder: " << sim->getOutputFolder() << std::endl;
        }

        // qDebug() << "\nType name for 65537:" << QMetaType::typeName(65537);
    } catch (const std::exception& e) {
        ErrorHandler::showError(e.what());
        return 1;
    }
    return 0;
}
