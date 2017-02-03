import nengo
import nengo.spa as spa
from nengo_extras.vision import Gabor, Mask

import pytry

import os
import itertools
import numpy as np
import png

verbose = True
fixation_time = 200 #ms

cur_path = '.'

D = 256
Dmid = 48
Dlow = 32


#load stimuli, subj=0 means a subset of the stims of subject 1 (no long words), works well with lower dims
#short=True does the same, except for any random subject, odd subjects get short words, even subjects long words
def load_stims(subj=0,short=True):

    #subj=0 is old, new version makes subj 0 subj 1 + short, but with a fixed stim set
    sub0 = False
    if subj==0:
        sub0 = True
        subj = 1
        short = True

    #pairs and words in experiment for training model
    global target_pairs #learned word pairs
    global target_words #learned words
    global items #all items in exp (incl. new foils)
    global rpfoil_pairs #presented rp_foils
    global newfoil_pairs #presented new foils

    #stimuli for running experiment
    global stims #stimuli in experiment
    global stims_target_rpfoils #target re-paired foils stimuli
    global stims_new_foils #new foil stimuli

    #load files (targets/re-paired foils; short new foils; long new foils)
    #ugly, but this way we can use the original stimulus files
    stims = np.genfromtxt(cur_path + '/stims/S' + str(subj) + 'Other.txt', skip_header=True,
                          dtype=[('probe', np.str_, 8), ('fan', 'int'), ('wordlen', np.str_, 8),
                                 ('item1', np.str_, 8), ('item2', np.str_, 8)], usecols=[1,2,3,4,5])
    stimsNFshort = np.genfromtxt(cur_path + '/stims/S' + str(subj) + 'NFShort.txt', skip_header=True,
                                 dtype=[('probe', np.str_, 8), ('fan', 'int'), ('wordlen', np.str_, 8),
                                        ('item1', np.str_, 8), ('item2', np.str_, 8)], usecols=[1,2,3,4,5])
    stimsNFlong = np.genfromtxt(cur_path + '/stims/S' + str(subj) + 'NFLong.txt', skip_header=True,
                                dtype=[('probe', np.str_, 8), ('fan', 'int'), ('wordlen', np.str_, 8),
                                       ('item1', np.str_, 8), ('item2', np.str_, 8)], usecols=[1,2,3,4,5])

    #if short, use a small set of new foils
    if short:
        if subj % 2 == 0: #even -> long
            stimsNF = random.sample(stimsNFlong,8)
        else: #odd -> short
            if sub0:  # not random when sub0 == True
                stimsNF = stimsNFshort[0:8]
            else:
                stimsNF = random.sample(stimsNFshort,8)
    else:
        stimsNF = np.hstack((stimsNFshort,stimsNFlong))

    #combine
    stims = np.hstack((stims, stimsNF))
    stims = stims.tolist()

    #if short, only keep shorts for odd subjects or longs for even subjects
    new_stims = []
    if short:
        for i in stims:
            if subj % 2 == 0 and i[2] == 'Long':
                new_stims.append(i)
            elif subj % 2 == 1 and i[2] == 'Short':
                new_stims.append(i)
        stims = new_stims

    #parse out different categories
    target_pairs = []
    rpfoil_pairs = []
    newfoil_pairs = []
    target_words = []
    stims_target_rpfoils = []
    stims_new_foils = []
    items = []

    for i in stims:

        # fill items list with all words
        items.append(i[3])
        items.append(i[4])

        # get target pairs
        if i[0] == 'Target':
            target_pairs.append((i[3],i[4]))
            target_words.append(i[3])
            target_words.append(i[4])
        elif i[0] == 'RPFoil':
            rpfoil_pairs.append((i[3],i[4]))
            target_words.append(i[3])
            target_words.append(i[4])
        else:
            newfoil_pairs.append((i[3],i[4]))


        # make separate lists for targets/rp foils and new foils (for presenting experiment)
        if i[0] != 'NewFoil':
            stims_target_rpfoils.append(i)
        else:
            stims_new_foils.append(i)

    # remove duplicates
    items = np.unique(items).tolist()
    #items.append('FIXATION')
    target_words = np.unique(target_words).tolist()



# load images for vision
def load_images():

    global X_train, y_train, y_train_words, fixation_image

    indir = cur_path + '/images/'
    files = os.listdir(indir)
    files2 = []

    #select only images for current item set
    for fn in files:
        if fn[-4:] == '.png' and ((fn[:-4] in items)):
             files2.append(fn)

    X_train = np.empty(shape=(np.size(files2), 90*14),dtype='float32') #images x pixels matrix
    y_train_words = [] #words represented in X_train
    for i,fn in enumerate(files2):
            y_train_words.append(fn[:-4]) #add word

            #read in image and convert to 0-1 vector
            r = png.Reader(indir + fn)
            r = r.asDirect()
            image_2d = np.vstack(itertools.imap(np.uint8, r[2]))
            image_2d /= 255
            image_1d = image_2d.reshape(1,90*14)
            X_train[i] = image_1d

    #numeric labels for words (could present multiple different examples of words, would get same label)
    y_train = np.asarray(range(0,len(np.unique(y_train_words))))
    X_train = 2 * X_train - 1  # normalize to -1 to 1


    #add fixation separately (only for presenting, no mapping to concepts)
    r = png.Reader(cur_path + '/images/FIXATION.png')
    r = r.asDirect()
    image_2d = np.vstack(itertools.imap(np.uint8, r[2]))
    image_2d /= 255
    fixation_image = np.empty(shape=(1,90*14),dtype='float32')
    fixation_image[0] = image_2d.reshape(1, 90 * 14)

#returns pixels of image representing item (ie METAL)
def get_image(item):
    if item != 'FIXATION':
        return X_train[y_train_words.index(item)]
    else:
        return fixation_image[0]



#### MODEL FUNCTIONS #####


# performs all steps in model ini
def initialize_model(subj=0,short=True):

    #warn when loading full stim set with low dimensions:
    if not(short) and not(high_dims):
        warn = warnings.warn('Initializing model with full stimulus set, but using low dimensions for vocabs.')

    load_stims(subj,short=short)
    load_images()
    initialize_vocabs()


#initialize vocabs
def initialize_vocabs():

    print('---- INITIALIZING VOCABS ----')

    global vocab_vision #low level visual vocab
    global vocab_concepts #vocab with all concepts and pairs
    global vocab_learned_words #vocab with learned words
    global vocab_all_words #vocab with all words
    global vocab_learned_pairs #vocab with learned pairs
    global vocab_motor #upper motor hierarchy (LEFT, INDEX)
    global vocab_fingers #finger activation (L1, R2)
    global vocab_goal #goal vocab
    global vocab_attend #attention vocab
    global vocab_reset #reset vocab

    global train_targets #vector targets to train X_train on
    global vision_mapping #mapping between visual representations and concepts
    global list_of_pairs #list of pairs in form 'METAL_SPARK'
    global motor_mapping #mapping between higher and lower motor areas


    #low level visual representations
    vocab_vision = nengo.spa.Vocabulary(Dmid,max_similarity=.5)
    for name in y_train_words:
        vocab_vision.parse(name)
    train_targets = vocab_vision.vectors


    #word concepts - has all concepts, including new foils, and pairs
    vocab_concepts = spa.Vocabulary(D, max_similarity=0.5)
    for i in y_train_words:
        vocab_concepts.parse(i)
    vocab_concepts.parse('ITEM1')
    vocab_concepts.parse('ITEM2')
    vocab_concepts.parse('NONE')

    list_of_pairs = []
    for item1, item2 in target_pairs:
        vocab_concepts.add('%s_%s' % (item1, item2), vocab_concepts.parse(
            '%s*ITEM1 + %s*ITEM2' % (item1, item2)))  # add pairs to concepts to use same vectors
        list_of_pairs.append('%s_%s' % (item1, item2))  # keep list of pairs notation

    # add all presented pairs to concepts for display
    for item1, item2 in newfoil_pairs:
        vocab_concepts.add('%s_%s' % (item1, item2), vocab_concepts.parse(
            '%s*ITEM1 + %s*ITEM2' % (item1, item2)))  # add pairs to concepts to use same vectors
    for item1, item2 in rpfoil_pairs:
        vocab_concepts.add('%s_%s' % (item1, item2), vocab_concepts.parse(
            '%s*ITEM1 + %s*ITEM2' % (item1, item2)))  # add pairs to concepts to use same vectors


    #vision-concept mapping between vectors
    vision_mapping = np.zeros((D, Dmid))
    for word in y_train_words:
        vision_mapping += np.outer(vocab_vision.parse(word).v, vocab_concepts.parse(word).v).T

    #vocab with learned words
    vocab_learned_words = vocab_concepts.create_subset(target_words + ['NONE'])

    #vocab with all words
    vocab_all_words = vocab_concepts.create_subset(y_train_words + ['ITEM1', 'ITEM2'])

    #vocab with learned pairs
    vocab_learned_pairs = vocab_concepts.create_subset(list_of_pairs) #get only pairs

    #print vocab_learned_words.keys
    #print y_train_words
    #print target_words

    #motor vocabs, just for sim calcs
    vocab_motor = spa.Vocabulary(Dmid) #different dimension to be sure, upper motor hierarchy
    vocab_motor.parse('LEFT+RIGHT+INDEX+MIDDLE')

    vocab_fingers = spa.Vocabulary(Dlow) #direct finger activation
    vocab_fingers.parse('L1+L2+R1+R2')

    #map higher and lower motor
    motor_mapping = np.zeros((Dlow, Dmid))
    motor_mapping += np.outer(vocab_motor.parse('LEFT+INDEX').v, vocab_fingers.parse('L1').v).T
    motor_mapping += np.outer(vocab_motor.parse('LEFT+MIDDLE').v, vocab_fingers.parse('L2').v).T
    motor_mapping += np.outer(vocab_motor.parse('RIGHT+INDEX').v, vocab_fingers.parse('R1').v).T
    motor_mapping += np.outer(vocab_motor.parse('RIGHT+MIDDLE').v, vocab_fingers.parse('R2').v).T

    #goal vocab
    vocab_goal = spa.Vocabulary(Dlow)
    vocab_goal.parse('DO_TASK')
    vocab_goal.parse('RECOG')
    vocab_goal.parse('RECOG2')
    vocab_goal.parse('FAMILIARITY')
    #vocab_goal.parse('RESPOND')
    #vocab_goal.parse('END')

    #attend vocab
    vocab_attend = vocab_concepts.create_subset(['ITEM1', 'ITEM2'])

    #reset vocab
    vocab_reset = spa.Vocabulary(Dlow)
    vocab_reset.parse('CLEAR+GO')



# word presented in current trial
global cur_item1
global cur_item2
cur_item1 = 'METAL' #just for ini
cur_item2 = 'SPARK'

# returns images of current words for display # fixation for 51 ms.
def present_pair(t):
    if t < (fixation_time/1000.0)+.002:
        return np.hstack((np.ones(7*90),get_image('FIXATION'),np.ones(7*90)))
    else:
        im1 = get_image(cur_item1)
        im2 = get_image(cur_item2)
        return np.hstack((im1, im2))


# returns image 1 <100 ms, otherwise image 2 || NOT USED ANYMORE
#def present_item(t):
#    if t < .1:
#        #print(cur_item1)
#        return get_image(cur_item1)
#    else:
#        #print(cur_item2)
#        return get_image(cur_item2)



def present_item2(t, output_attend):

    #no-attention scale factor
    no_attend = .1

    #first fixation before start trial
    if t < (fixation_time/1000.0) + .002:
        # ret_ima = np.zeros(1260)
        ret_ima = no_attend * get_image('FIXATION')
    else: #then either word or zeros (mix of words?)
        attn = vocab_attend.dot(output_attend) #dot product with current input (ie ITEM 1 or 2)
        i = np.argmax(attn) #index of max

        #ret_ima = np.zeros(1260)

        ret_ima = no_attend * (get_image(cur_item1) + get_image(cur_item2)) / 2

        if attn[i] > 0.3: #if we really attend something
            if i == 0: #first item
                ret_ima = get_image(cur_item1)
            else:
                ret_ima = get_image(cur_item2)

    return (.8 * ret_ima)


#initialize model
def create_model(p):
    model = spa.SPA()
    with model:

        # control
        model.control_net = nengo.Network()
        with model.control_net:

            model.attend = spa.State(D, vocab=vocab_attend, feedback=.5)  # vocab_attend
            model.goal = spa.State(Dlow, vocab=vocab_goal, feedback=.7)  # current goal


        ### vision ###

        # set up network parameters
        n_vis = X_train.shape[1]  # nr of pixels, dimensions of network
        n_hid = 1000  # nr of gabor encoders/neurons

        # random state to start
        rng = np.random.RandomState(9)
        encoders = Gabor().generate(n_hid, (4, 4), rng=rng)  # gabor encoders, 11x11 apparently, why?
        encoders = Mask((14, 90)).populate(encoders, rng=rng,
                                           flatten=True)  # use them on part of the image


        model.visual_net = nengo.Network()
        with model.visual_net:

            #represent currently attended item
            model.attended_item = nengo.Node(present_item2,size_in=D)
            nengo.Connection(model.attend.output, model.attended_item)

            model.vision_gabor = nengo.Ensemble(n_hid, n_vis, eval_points=X_train,
                                                    neuron_type=nengo.AdaptiveLIF(inc_n=0.05, tau_n=0.01),
                                                    #neuron_type=nengo.LIF(),
                                                    intercepts=nengo.dists.Uniform(-0.1, 0.1),
                                                    #intercepts=nengo.dists.Choice([-0.5]), #should we comment this out? not sure what's happening
                                                    #max_rates=nengo.dists.Choice([100]),
                                                    encoders=encoders)
            #recurrent connection (time constant 500 ms)
            # strength = 1 - (100/500) = .8

            zeros = np.zeros_like(X_train)
            nengo.Connection(model.vision_gabor, model.vision_gabor, synapse=0.005, #.1
                             eval_points=np.vstack([X_train, zeros, np.random.randn(*X_train.shape)]),
                             transform=0.5)

            model.visual_representation = nengo.Ensemble(n_hid, dimensions=Dmid)

            model.visconn = nengo.Connection(model.vision_gabor, model.visual_representation, synapse=0.005, #was .005
                                            eval_points=X_train, function=train_targets,
                                            solver=nengo.solvers.LstsqL2(reg=0.01))
            nengo.Connection(model.attended_item, model.vision_gabor, synapse=.02) #.03) #synapse?

            #model.visconn_recurr = nengo.Connection(model.visual_representation, model.vision_gabor,
            #                                synapse=0.01, #was .005
            #                                eval_points=train_targets,
            #                                function=X_train,
            #                                solver=nengo.solvers.LstsqL2(reg=0.01),
            #                                transform=1.0)


        ### central cognition ###

        # concepts
        model.concepts = spa.AssociativeMemory(vocab_all_words, #vocab_concepts,
                                               wta_output=True,
                                               wta_inhibit_scale=1, #was 1
                                               #default_output_key='NONE', #what to say if input doesn't match
                                               threshold=0.3)  # how strong does input need to be for it to recognize
        nengo.Connection(model.visual_representation, model.concepts.input, transform=.8*vision_mapping) #not too fast to concepts, might have to be increased to have model react faster to first word.

        #concepts accumulator
        model.concepts_evidence = spa.State(1, feedback=1, feedback_synapse=0.005) #the lower the synapse, the faster it accumulates (was .1)
        concepts_evidence_scale = 2.5
        nengo.Connection(model.concepts.am.elem_output, model.concepts_evidence.input,
                         transform=concepts_evidence_scale * np.ones((1, model.concepts.am.elem_output.size_out)),synapse=0.005)

        #concepts switch
        model.do_concepts = spa.AssociativeMemory(vocab_reset, default_output_key='CLEAR', threshold=.2)
        nengo.Connection(model.do_concepts.am.ensembles[-1], model.concepts_evidence.all_ensembles[0].neurons,
                         transform=np.ones((model.concepts_evidence.all_ensembles[0].n_neurons, 1)) * -10,
                         synapse=0.005)

        # pair representation
        model.vis_pair = spa.State(D, vocab=vocab_all_words, feedback=1.0, feedback_synapse=.05) #was 2, 1.6 works ok, but everything gets activated.


        actions = dict(
            #wait & start
            a_aa_wait =          'dot(goal,WAIT) - .9 --> goal=0',

            f_judge_familiarity = 'dot(goal, FAMILIARITY) - 0.1'
                                  '--> goal=FAMILIARITY',

            )


        if not p.skip_action_b:
            actions.update(dict(
                a_attend_item1    =  'dot(goal,DO_TASK) - .1 '
                                     '--> goal=RECOG, attend=ITEM1, '
                                         'do_concepts=GO',

                b_attending_item1 =  'dot(goal,RECOG) + '
                                       'dot(attend,ITEM1) - '
                                       'concepts_evidence - .3 '
                                       '--> goal=RECOG, attend=ITEM1,'
                                           'do_concepts=GO ',
                ))
        else:
            actions.update(dict(
                a_attend_item1    =  'dot(goal,DO_TASK) - .1 '
                                     '--> goal=RECOG-DO_TASK, attend=ITEM1, '
                                         'do_concepts=GO',
                ))


        if not p.skip_action_d:
            actions.update(dict(
                c_attend_item2    =  'dot(goal,RECOG) + dot(attend,ITEM1) +'
                                     'concepts_evidence - 1.6 '
                                     '--> goal=RECOG2, attend=ITEM2,'
                                         'vis_pair=%g*(ITEM1*concepts)'
                                         '' % p.item1_weight,

                d_attending_item2 =  'dot(goal,RECOG2+RECOG) + '
                                     ' dot(attend,ITEM2) - '
                                     ' concepts_evidence - .4 '
                                     '--> goal=RECOG2-RECOG,'
                                         'attend=ITEM2, '
                                         'do_concepts=GO ',

                e_judge_familiarity =  'dot(goal,RECOG2) + '
                                       ' dot(attend,ITEM2) +'
                                       ' concepts_evidence - 1.8 '
                                       '--> goal=FAMILIARITY, '
                                       '    vis_pair=%g*(ITEM2*concepts)'
                                       '' % p.item2_weight,
                ))
        else:
            actions.update(dict(
                c_attend_item2    =  'dot(goal,RECOG) + dot(attend,ITEM1) +'
                                     'concepts_evidence - 1.6 '
                                     '--> goal=RECOG2-RECOG, attend=ITEM2,'
                                         'do_concepts=GO, '
                                         'vis_pair=%g*(ITEM1*concepts)'
                                         '' % p.item1_weight,

                e_judge_familiarity =  'dot(goal,RECOG2) + '
                                       ' dot(attend,ITEM2) +'
                                       ' concepts_evidence - 1.8 '
                                       '--> goal=FAMILIARITY, '
                                       '    attend=ITEM2, '
                                       '    vis_pair=%g*(ITEM2*concepts)'
                                       '' % p.item2_weight,
            ))



        model.bg = spa.BasalGanglia(
            spa.Actions(**actions))

        model.thalamus = spa.Thalamus(model.bg)

        #probes
        #model.pr_motor_pos = nengo.Probe(model.finger_pos.output,synapse=.01) #raw vector (dimensions x time)
        #model.pr_motor = nengo.Probe(model.fingers.output,synapse=.01)
        #model.pr_motor1 = nengo.Probe(model.motor.output, synapse=.01)

        model.pr_vision_gabor = nengo.Probe(model.vision_gabor.neurons,
                                            synapse=0.005) #do we need synapse, or should we do something with the spikes
        #model.pr_familiarity = nengo.Probe(model.dm_learned_words.am.elem_output,synapse=.01) #element output, don't include default
        #model.pr_concepts = nengo.Probe(model.concepts.am.elem_output, synapse=.01)  # element output, don't include default

        #multiply spikes with the connection weights

        model.pr_vis_pair = nengo.Probe(model.vis_pair.output, synapse=0.03)
        model.pr_goal = nengo.Probe(model.goal.output, synapse=0.03)
        model.pr_thal = nengo.Probe(model.thalamus.output, synapse=0.01)
        model.pr_ce = nengo.Probe(model.concepts_evidence.output, synapse=0.01)

        model.action_names = [a.name for a in model.bg.actions.actions]


        #input
        model.input = spa.Input(goal=goal_func)



        #print(sum(ens.n_neurons for ens in model.all_ensembles))

    return model



def goal_func(t):
    if t < (fixation_time/1000.0) + .002:
        return 'WAIT'
    elif t < (fixation_time/1000.0) + .022: #perhaps get from distri
        return 'DO_TASK'
    else:
        return '0'  # first 50 ms fixation


cur_item1 = 'METAL'
cur_item2 = 'SPARK'


class ExploreVision(pytry.NengoTrial):
    def params(self):
        self.param('time to run', T=0.5)
        self.param('skip action b', skip_action_b=False)
        self.param('skip action d', skip_action_d=False)
        self.param('weight for item 1', item1_weight=3.0)
        self.param('weight for item 2', item2_weight=1.9)

    def model(self, p):
        initialize_model(subj=0)
        model = create_model(p=p)

        self.pr_vision_gabor = model.pr_vision_gabor
        self.pr_vis_pair = model.pr_vis_pair
        self.pr_goal = model.pr_goal
        self.pr_thal = model.pr_thal
        self.action_names = model.action_names
        self.pr_ce = model.pr_ce
        return model

    def evaluate(self, p, sim, plt):
        sim.run(p.T)
        vis_activ = np.sum(sim.data[self.pr_vision_gabor], axis=1)

        vp_item1 = np.dot(sim.data[self.pr_vis_pair],
                          vocab_all_words.parse('%s*ITEM1' % cur_item1).v)
        vp_item2 = np.dot(sim.data[self.pr_vis_pair],
                          vocab_all_words.parse('%s*ITEM2' % cur_item2).v)

        goal = np.dot(sim.data[self.pr_goal],
                      vocab_goal.vectors.T)

        if plt is not None:
            plt.subplot(5, 1, 1)
            plt.plot(sim.trange(), vis_activ)
            plt.ylabel('visual activity')
            plt.subplot(5, 1, 2)
            plt.plot(sim.trange(), vp_item1, label=cur_item1)
            plt.plot(sim.trange(), vp_item2, label=cur_item2)
            plt.legend(loc='best')
            plt.ylabel('vis_pair')
            plt.subplot(5, 1, 3)
            plt.plot(sim.trange(), goal, linewidth=3)
            plt.legend(vocab_goal.keys, loc='upper left', fontsize='x-small')
            plt.ylabel('goal')
            plt.subplot(5, 1, 4)
            plt.plot(sim.trange(), sim.data[self.pr_ce])
            plt.ylabel('concepts_evidence')
            plt.subplot(5, 1, 5)
            plt.plot(sim.trange(), sim.data[self.pr_thal], linewidth=3)
            plt.legend(self.action_names, loc='upper left', fontsize='x-small')
            plt.ylabel('action')
        return {}
