use crate::agent::*;

pub fn assign(input: &mut [Neuron; INPS]) {
    for (i, res) in csv::Reader::from_reader(std::io::stdin()).records().enumerate() {
        input[i].excitation = res.unwrap()[0].parse().unwrap() // placeholder
    }
}
