function make_packets(packet_size::Int, sample::Vector)
  sample_size = length(sample)
  [ sample[i:i+packet_size-1] for i in 1:packet_size:sample_size ]
end

function packetwise_mean(packet_size::Int, estimator::Function, sample::Vector)
  packets = make_packets(packet_size, sample)
  ests = map(estimator, packets)
  mean(ests)
end
