#include "TChain.h"
#include "TH1F.h"
#include "TLorentzVector.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TRootLHEFParticle.h"
#include "ExRootClasses.h"
#include "ExRootTreeReader.h"
#include "TClonesArray.h"

void ana()
{
    // Load the LHEF events file into a TChain
    TChain chain("LHEF");
    chain.Add("../hww_cHW1/Events/events_cHW10.000000/unweighted_events.root");

    // Open a file to save the output
    TFile *fout = new TFile("output2_bsm_10.root", "RECREATE");

    // Create an NTuple to store results with all the new features
    TNtuple *ntuple = new TNtuple("eventData", "Event Data with Features", 
        "l0_id:l1_id:q0_id:q1_id:"
        "l0_pt:l1_pt:q0_pt:q1_pt:"
        "l0_phi:l1_phi:q0_phi:q1_phi:"
        "l0_eta:l1_eta:q0_eta:q1_eta:"
        "l0_m:l1_m:q0_m:q1_m:"
        "l0_e:l1_e:q0_e:q1_e:"
        "met_et:met_phi:m_ll:m_qq:"
        "pt_ll:pt_qq:d_phi_ll:d_phi_qq:"
        "d_eta_ll:d_eta_qq:d_y_ll:d_y_qq:"
        "sqrtHT:MET_sig:m_l0q0:m_l0q1:m_l1q0:m_l1q1");

    // Create ExRootTreeReader to process the events
    ExRootTreeReader *treeReader = new ExRootTreeReader(&chain);
    Long64_t numberOfEntries = treeReader->GetEntries();

    // Access the "Particle" branch
    TClonesArray *branchParticle = treeReader->UseBranch("Particle");

    // Loop over all events in the file
    for(Long64_t entry = 0; entry < numberOfEntries; ++entry)
    {
        // Read event data
        treeReader->ReadEntry(entry);
        if (entry % 100000 == 0) cout << "Processing Event " << entry << endl;

        // Initialize variables to store particles and their IDs
        TLorentzVector electron, positron;
        int electron_id = 0, positron_id = 0;
        TLorentzVector upQuark, downQuark;
        int upQuark_id = 0, downQuark_id = 0;

        // Variables for MET
        double met_px = 0.0;
        double met_py = 0.0;

        // Loop over particles in the event
        for(Int_t part_i = 0; part_i < branchParticle->GetEntries(); ++part_i)
        {
            TRootLHEFParticle *particle = (TRootLHEFParticle*) branchParticle->At(part_i);

            // Electrons and positrons
            if(particle->PID == 11) // Electron
            {
                electron.SetPtEtaPhiE(particle->PT, particle->Eta, particle->Phi, particle->E);
                electron_id = particle->PID;
            }
            else if(particle->PID == -11) // Positron
            {
                positron.SetPtEtaPhiE(particle->PT, particle->Eta, particle->Phi, particle->E);
                positron_id = particle->PID;
            }

            // Up and down quarks
            else if(particle->PID == 2) // Up quark
            {
                upQuark.SetPtEtaPhiE(particle->PT, particle->Eta, particle->Phi, particle->E);
                upQuark_id = particle->PID;
            }
            else if(particle->PID == 1) // Down quark
            {
                downQuark.SetPtEtaPhiE(particle->PT, particle->Eta, particle->Phi, particle->E);
                downQuark_id = particle->PID;
            }

            // Neutrinos contribute to MET
            else if(abs(particle->PID) == 12 || abs(particle->PID) == 14 || abs(particle->PID) == 16) // Neutrinos
            {
                met_px += particle->PT * cos(particle->Phi);
                met_py += particle->PT * sin(particle->Phi);
            }
        }

        // Calculate MET
        double met_et = sqrt(met_px * met_px + met_py * met_py);
        double met_phi = atan2(met_py, met_px);

        // Check if all particles were found
        if(electron.Pt() > 0 && positron.Pt() > 0 && upQuark.Pt() > 0 && downQuark.Pt() > 0)
        {
            // Determine the most energetic lepton
            TLorentzVector l0, l1;
            int l0_id, l1_id;

            if(electron.Pt() > positron.Pt())
            {
                l0 = electron;
                l0_id = electron_id;
                l1 = positron;
                l1_id = positron_id;
            }
            else
            {
                l0 = positron;
                l0_id = positron_id;
                l1 = electron;
                l1_id = electron_id;
            }

            // Determine the most energetic quark
            TLorentzVector q0, q1;
            int q0_id, q1_id;

            if(upQuark.Pt() > downQuark.Pt())
            {
                q0 = upQuark;
                q0_id = upQuark_id;
                q1 = downQuark;
                q1_id = downQuark_id;
            }
            else
            {
                q0 = downQuark;
                q0_id = downQuark_id;
                q1 = upQuark;
                q1_id = upQuark_id;
            }

            // Transverse momentum
            double l0_pt = l0.Pt();
            double l1_pt = l1.Pt();
            double q0_pt = q0.Pt();
            double q1_pt = q1.Pt();

            // Phi
            double l0_phi = l0.Phi();
            double l1_phi = l1.Phi();
            double q0_phi = q0.Phi();
            double q1_phi = q1.Phi();

            // Eta
            double l0_eta = l0.Eta();
            double l1_eta = l1.Eta();
            double q0_eta = q0.Eta();
            double q1_eta = q1.Eta();

            // Mass
            double l0_m = l0.M();
            double l1_m = l1.M();
            double q0_m = q0.M();
            double q1_m = q1.M();

            // Energy
            double l0_e = l0.E();
            double l1_e = l1.E();
            double q0_e = q0.E();
            double q1_e = q1.E();

            // Center-of-mass energies
            double m_ll = (l0 + l1).M();
            double m_qq = (q0 + q1).M();

            // Transverse momentum of systems
            double pt_ll = (l0 + l1).Pt();
            double pt_qq = (q0 + q1).Pt();

            // Angle differences
            double d_phi_ll = l0.DeltaPhi(l1);
            double d_phi_qq = q0.DeltaPhi(q1);

            // Pseudorapidity differences
            double d_eta_ll = l0.Eta() - l1.Eta();
            double d_eta_qq = q0.Eta() - q1.Eta();

            // Rapidity differences
            double d_y_ll = l0.Rapidity() - l1.Rapidity();
            double d_y_qq = q0.Rapidity() - q1.Rapidity();

            // sqrtHT
            double sqrtHT = sqrt(l0_pt + l1_pt + q0_pt + q1_pt + met_et);

            // MET significance
            double MET_sig = met_et / sqrtHT;

            // Center-of-mass energies for lepton-quark pairs
            double m_l0q0 = (l0 + q0).M();
            double m_l0q1 = (l0 + q1).M();
            double m_l1q0 = (l1 + q0).M();
            double m_l1q1 = (l1 + q1).M();

            // Fill the NTuple
            ntuple->Fill(
                l0_id, l1_id, q0_id, q1_id,
                l0_pt, l1_pt, q0_pt, q1_pt,
                l0_phi, l1_phi, q0_phi, q1_phi,
                l0_eta, l1_eta, q0_eta, q1_eta,
                l0_m, l1_m, q0_m, q1_m,
                l0_e, l1_e, q0_e, q1_e,
                met_et, met_phi, m_ll, m_qq,
                pt_ll, pt_qq, d_phi_ll, d_phi_qq,
                d_eta_ll, d_eta_qq, d_y_ll, d_y_qq,
                sqrtHT, MET_sig, m_l0q0, m_l0q1, m_l1q0, m_l1q1
            );
        }
    }

    // Save the results to the output file
    fout->cd();
    ntuple->Write();
    fout->Close();
}