?	?t???@?t???@!?t???@	o٤F?i@o٤F?i@!o٤F?i@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?t???@ffffff??Au?V??@Y?G?z?D@*	    b?)A2?
TIterator::Model::MaxIntraOpParallelism::Map::Prefetch::BatchV2::Shuffle::Zip[0]::Map?㥛? y?@!?nb??XW@)?"??~q?@1N??QW@:Preprocessing2b
+Iterator::Model::MaxIntraOpParallelism::Map???Mb?D@!"b?ӆ?@)`??"??D@1??ЉL?@:Preprocessing2u
>Iterator::Model::MaxIntraOpParallelism::Map::Prefetch::BatchV2??x?&?U@!?cNAm?$@)?x?&1H#@1S???e??:Preprocessing2?
LIterator::Model::MaxIntraOpParallelism::Map::Prefetch::BatchV2::Shuffle::Zip??x?&1??@!??Z?qW@)?C?l?? @1?lr>?3??:Preprocessing2?
\Iterator::Model::MaxIntraOpParallelism::Map::Prefetch::BatchV2::Shuffle::Zip[1]::TensorSlice?=
ףp=??!???ϡf??)=
ףp=??1???ϡf??:Preprocessing2?
aIterator::Model::MaxIntraOpParallelism::Map::Prefetch::BatchV2::Shuffle::Zip[0]::Map::TensorSlice??I+???!?p?????)?I+???1?p?????:Preprocessing2
GIterator::Model::MaxIntraOpParallelism::Map::Prefetch::BatchV2::Shuffle???Q?ǁ@!?,?P??P@)?p=
ף??10?.????:Preprocessing2l
5Iterator::Model::MaxIntraOpParallelism::Map::Prefetch?&1???!???Q(?x?)?&1???1???Q(?x?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism/?$?D@!G?n!#?@){?G?zt?1??0??C?:Preprocessing2F
Iterator::Model??K7??D@!eUC,??@)????Mbp?1Is?B??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9o٤F?i@Ii??[g?W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ffffff??ffffff??!ffffff??      ??!       "      ??!       *      ??!       2	u?V??@u?V??@!u?V??@:      ??!       B      ??!       J	?G?z?D@?G?z?D@!?G?z?D@R      ??!       Z	?G?z?D@?G?z?D@!?G?z?D@b      ??!       JCPU_ONLYYo٤F?i@b qi??[g?W@Y      Y@q|??^?c?"?
device?Your program is NOT input-bound because only 4.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 