# Random Cut Forest

This directory contains a Rust implementation, that mirrors the Java implementation 
of the Random Cut Forest (RCF).

The compact trees in the Java version of RCF2.0 was designed with memory 
safety in mind. This rust implementation skips over that version and mirrors Java RCF3.0.

Rust provides memory safety and the parallel implementation of the same algorithm in different 
languages allows us to get a (qualified) verification of safety. At the same time, verifying the 
randomness of randomized data structure is non-trivial, and the existing tests of the Java version 
provide a qualified verification. We expect the different 
implementations to remain in sync. 



