import time

class rock_paper_scissors:
    def __init__(self, MIN=0, MAX=0.012, velocity=0.001):
        self.fingers_state = {'index_control' : None,
                             'middle_control' : None,
                             'ring_control' : None,
                             'little_control' : None,
                             'thumb_joint_control' : None,
                             'thumb_control' : None
                            }
        self.reset()
                
        self.MIN = MIN
        self.MAX = MAX
        self.velocity = velocity

    def __call__(self, hand):
        self.rps(hand)
        return self.fingers_state

    def rps(self, hand):
        if hand == 0:
            self.normal()
        elif hand == 1 or hand == 2:
            self.rock()
        elif hand == 3 or hand == 4:
            self.scissors()
        else:
            self.paper()

    def reset(self):
        for i, m in self.fingers_state.items():
            self.fingers_state[i] = 0
    
    def normal(self):
        for i, m in self.fingers_state.items():
            if self.fingers_state[i] > self.MIN:
                self.fingers_state[i] -= self.velocity
    
    def rock(self):
        for i, m in self.fingers_state.items():
            if self.fingers_state[i] < self.MAX:
                self.fingers_state[i] += self.velocity

    def paper(self):
        for i, m in self.fingers_state.items():
            if self.fingers_state[i] > self.MIN:
                self.fingers_state[i] -= self.velocity

    def scissors(self):
        for i, m in self.fingers_state.items():
            if i == 'index_control' or i == 'middle_control':
                if self.fingers_state[i] > self.MIN:
                    self.fingers_state[i] -= self.velocity
            else:
                if self.fingers_state[i] < self.MAX:
                    self.fingers_state[i] += self.velocity


class control:
    def __init__(self, hj_tf, sleep_time = 0.01):
        self.hj_tf = hj_tf
        self.stm_index_id = 0x01
        self.stm_middle_id = 0x02
        self.stm_ring_id = 0x03
        self.stm_little_id = 0x04
        self.stm_thumb_id = 0x05
        self.stm_thumb_joint_id = 0x06
        self.sleep_time = sleep_time

    def move(self, fingers_state):
        index_pris_val = fingers_state['index_control']
        middle_pris_val = fingers_state['middle_control']
        ring_pris_val = fingers_state['ring_control']
        little_pris_val = fingers_state['little_control']
        thumb_pris_val = fingers_state['thumb_joint_control']
        thumb_joint_pris_val = fingers_state['thumb_control']

        # index
        self.hj_tf.index_control(index_pris_val)
        self.hj_tf.hj_finger_control(self.stm_index_id, index_pris_val)

        # middle
        self.hj_tf.middle_control(middle_pris_val)
        self.hj_tf.hj_finger_control(self.stm_middle_id, middle_pris_val)

        # ring
        self.hj_tf.ring_control(ring_pris_val)
        self.hj_tf.hj_finger_control(self.stm_ring_id, ring_pris_val)

        # little
        self.hj_tf.little_control(little_pris_val)
        self.hj_tf.hj_finger_control(self.stm_little_id, little_pris_val)

        # thumb_joint
        self.hj_tf.thumb_joint_control(thumb_joint_pris_val)
        self.hj_tf.hj_finger_control(self.stm_thumb_joint_id, thumb_joint_pris_val)

        # thumb
        self.hj_tf.thumb_control(thumb_pris_val)
        self.hj_tf.hj_finger_control(self.stm_thumb_id, thumb_pris_val)

        time.sleep(self.sleep_time)



