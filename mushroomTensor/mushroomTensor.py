import tensorflow as tf
import random

def make_oneshot_dic(input_list, input_val):
    #mySplit = string_input.split(",")

    container = [0]*len(input_list)
    if input_val in input_list:
        container[input_list.index(input_val)] = 1
    #for k, i in zip(input_list, range(len(input_list))):
     #   k = k.strip()
      #  oneShotList = [0]*len(input_list)
       # oneShotList[i] = 1
        #container[k] = oneShotList
    return container

#creates enumerated lists of the attributes to be compared to later
def make_enum_list(string_input):
    #mySplit, newList = make_oneshot_dic(string_input)
    mySplit = string_input.split(",")

    newList = [0]*len(mySplit)
    for k, i in enumerate(mySplit):
        newList[k]= k
        #print (newList)
    #zippedList = zip(mySplit, newList)
    #scale_Enum_list(newList)
    return mySplit, newList

def scale_Enum_list(enumList):
    scaleList = enumList
    scaleRange = len(enumList)

#parses the file and sorts the data
def parse_input_file(filename):

    #setting up lists of the attributes
    poisonous, Answer = make_enum_list("p,e")
    cap_shape, cap_shape1 = make_enum_list("b,c,x,f,k,s")
    cap_surface, cap_surface1 = make_enum_list("f,g,y,s")
    cap_color, cap_color1 = make_enum_list("n,b,c,g,r,p,u,e,w,y,t,f")
    bruises, bruises1 = make_enum_list("t,f")
    odor, odor1 = make_enum_list("a,l,c,y,f,m,n,p,s")
    gill_attachment, gill_attachment1 = make_enum_list("a,d,f,n")
    gill_spacing, gill_spacing1 = make_enum_list("c,w,d")
    gill_size, gill_size1 = make_enum_list("b,n")
    gill_color, gill_color1 = make_enum_list("k,n,b,h,g,r,o,p,u,e,w,y")
    stalk_shape, stalk_shape1 = make_enum_list("e,t")
    stalk_root, stalk_root1 = make_enum_list("b,c,u,e,z,r,?")
    stalk_surface_above_ring, stalk_surface_above_ring1 = make_enum_list("f,y,k,s")
    stalk_surface_below_ring, stalk_surface_below_ring1 = make_enum_list("f,y,k,s")
    stalk_color_above_ring, stalk_color_above_ring1 = make_enum_list("n,b,c,g,o,p,e,w,y")
    stalk_color_below_ring, stalk_color_below_ring1 = make_enum_list("n,b,c,g,o,p,e,w,y")
    veil_typ, veil_typ1 = make_enum_list("p,u")
    veil_color, veil_color1 = make_enum_list("n,o,w,y")
    ring_number, ring_number1 = make_enum_list("n,o,t")
    ring_typ, ring_typ1 = make_enum_list("c,e,f,l,n,p,s,z")
    spore_color, spore_color1 = make_enum_list("k,n,b,h,r,o,u,w,y")
    population, population1 = make_enum_list("a,c,n,s,v,y")
    habitat, habitat1 = make_enum_list("g,l,m,p,u,w,d")
    
    #The most jank as fuck fix to my stupidity
    containerData = [cap_shape,cap_surface,cap_color,bruises,odor,gill_attachment,gill_spacing,gill_size,gill_color,stalk_shape,stalk_root,stalk_surface_above_ring,stalk_surface_below_ring,stalk_color_above_ring,stalk_color_below_ring,veil_typ,veil_color,ring_number,ring_typ,spore_color,population,habitat]

    containerEnums = [cap_shape1,cap_surface1,cap_color1,bruises1,odor1,gill_attachment,gill_spacing1,gill_size1,gill_color1,stalk_shape1,stalk_root1,stalk_surface_above_ring1,stalk_surface_below_ring1,stalk_color_above_ring1,stalk_color_below_ring1,veil_typ1,veil_color1,ring_number1,ring_typ1,spore_color1,population1,habitat1]

    labels = []
    data = []

    with open(filename, "r") as file:
        for line in file:
            if (len(line.strip()) == 0):
                continue

            input_line = line.strip().split(",")

            lineLabel = input_line[0]
            dataLine = input_line[1:23]

            #print(lineLabel)
            #print (dataLine)

            if lineLabel in poisonous:
                lineLabel = Answer[poisonous.index(lineLabel)]
                #print(lineLabel)
                labels.append(lineLabel)
            else:
                print ("Bad answer input")
            
            for x in range(22):
                if dataLine[x] in containerData[x]:
                    #print(dataLine[x])
                    #print (containerData[x])
                    #charCheck = dataLine[x]
                    #print (charCheck)
                    #dataLine[x] = containerEnums[x][containerData[x].index(charCheck)]
                    dataToAdd = make_oneshot_dic(containerData[x],dataLine[x])
                    #print(dataToAdd)
                    data.append(dataToAdd)
                   
                else:
                    print ("Bad input data")
            
            
        return data, labels

def neural_net_initializer(filename):

    data , labels = parse_input_file(filename)
    dataSet = list(zip(data, labels))
    random.shuffle(dataSet)
    test_length = int(len(dataSet)*0.67)

    train_dataSet = dataSet[:test_length]
    test_dataSet = dataSet[test_length:]

    x_size = len(data)
    output_size = 2
    num_nodes = 200

    inputs = tf.placeholder("float", shape=[None, x_size])
    myLabels = tf.placeholder("int32", shape=[None])

    weights1 = tf.get_variable("weight1", shape=[x_size, num_nodes], initializer=tf.contrib.layers.xavier_initializer())
    bias1 = tf.get_variable("bias1", shape=[num_nodes], initializer=tf.constant_initializer(value=0.0))
    
    layer1 = tf.nn.relu(tf.matmul(inputs, weights1) + bias1)

    weights2 = tf.get_variable("weight2", shape=[num_nodes, num_nodes], initializer=tf.contrib.layers.xavier_initializer())
    bias2 = tf.get_variable("bias2", shape=[num_nodes], initializer=tf.constant_initializer(value=0.0))
    
    layer2 = tf.nn.relu(tf.matmul(layer1, weights2) + bias2)

    weights3 = tf.get_variable("weight3", shape=[num_nodes, output_size], initializer=tf.contrib.layers.xavier_initializer())
    bias3 = tf.get_variable("bias3", shape=[output_size], initializer=tf.constant_initializer(value=0.0))
    
    outputs = tf.matmul(layer2, weights3) + bias3

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(myLabels, output_size), logits=outputs))
    train = tf.train.AdamOptimizer().minimize(loss)

    predictions = tf.argmax(tf.nn.softmax(outputs), axis=1)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(5000):
            batch = random.sample(train_dataSet, 25)
            inputs_batch, labels_batch = zip(*batch)
            loss_output, prediction_output, _ = sess.run(
                [loss, predictions, train],
                feed_dict={inputs: inputs_batch,
                           labels: labels_batch})

            # print("Prediction output", prediction_output)
            # print("Labels batch", labels_batch)

            # accuracy = np.mean(labels_batch == prediction_output)

            # print("train", "loss", loss_output, "accuracy", accuracy)

        batch = random.sample(test_dataset, 100)
        inputs_batch, labels_batch = zip(*batch)
        loss_output, prediction_output, _ = sess.run(
            [loss, predictions, train],
            feed_dict={inputs: inputs_batch,
                       labels: labels_batch})
        accuracy = np.mean(labels_batch == prediction_output)

        f = open('output.txt', 'w')
        print("Prediction output: ", prediction_output)
        myString = 'Prediction output: ' + np.array_str(prediction_output)
        f.write(myString)
        f.write('\n\n')
        print("Labels batch: ", labels_batch)
        t = ' '.join(str(v) for v in labels_batch)
        myString = 'Labels batch:      [' + t[0:73] + '\n ' + t[74:147] + '\n ' + t[148:] + ']\n\n'
        f.write(myString)
        print("Loss: ", loss_output)
        myString = 'Loss: ' + np.array_str(loss_output)
        f.write(myString + '\n\n')
        print("Accuracy: ", accuracy)
        myString = 'Accuracy: ' + np.array_str(accuracy)
        f.write(myString)
        # I believe this is also training, how to run without training?
        f.close
        
if(__name__=="__main__"):
    #parse_input_file
    neural_net_initializer("mushroomData.txt")
