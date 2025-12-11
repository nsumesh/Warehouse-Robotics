import time
import math

class FSM:
    def __init__(self, node):
        self.node = node
        self.states = {'idle': self.idle_state, 'pickup': self.pickup_state, 'dropoff': self.dropoff_state, 'docking': self.docking_state}
    
    def move(self):
        self.node.cleaup_dropped_items()
        self.handle_collisions()
        self.task_timeout()
        state = self.states.get(self.node.phase)
        if state:
            state(time.time())
        else:
            self.node.get_logger().error("Unknown FSM state")
        
    def handle_collisions(self):
        node = self.node
        if node.phase not in ["pickup", "dropoff", "docking"]:
            node.last_collision_time = None
            return
        if node.start_time is None:
            node.start_time = time.time()
        if time.time() - node.start_time<2.0:
            return
        node.collision_check_enabled = True
        if node.collision_check_enabled and node.check_collision():
            curr = time.time()
            if node.last_collision_time is None:
                node.last_collision_time = curr
            elif curr-node.last_collision_time>3.0:
                node.get_logger.warn("Resetting position due to collision")
                node.reset_robot_position()
                node.task_start_time = curr
                node.last_collision_time = None
        else:
            node.last_collision_time = None
    
    def timeout(self, curr):
        node = self.node
        if node.task_start_time and curr-node.task_start_time>node.max_task_time:
            node.get_logger().warn("Moving to next phase")
            node.advance_phase()
    
    def idle_state(self, curr):
        node = self.node
        if not node.task_queue:
            return
        node.current_task = node.task_queue.pop(0)
        node.phase = 'pickup'
        node.task_start_time = curr
        node.current_goal = node.pickup_location
        node.goal_reached_time = None
        node.task = None
        node.items_spawned_for_current_task = False
        node.items_dropped_for_current_task = False
        pickup_x, pickup_y = node.pickup_location
        node.get_logger.info("Going to pickup for task "+str(node.current_task))
    
    def pickup_state(self, curr):
        node = self.node
        goal_distance = node.goal_distance()
        if goal_distance<1.5 and node.items_spawned_for_current_task==False:
            if node.spawn_items_for_task():
                node.items_spawned_for_current_task=True
        
        if node.goal_reached_check():
            node.phase = 'dropoff'
            node.task_start_time = curr
            node.task = node.current_task
            node.current_goal = node.drop_docks[node.task]
            node.goal_reached_time = None
            if node.items_at_pickup.get(node.current_task):
                item_id = node.items_at_pickup[node.current_task].pop(0)
            else:
                node.item_counter[node.current_task]+=1
                item_id = "item_" + str(node.current_task) + "_" + str(node.item_counter[node.current_task])
            node.current_item_id = item_id
            node.virtual_pickup(item_id. node.task)
            x,y = node.current_goal
            node.get_logger.info("Picked up "+item_id)
    
    def dropoff_state(self, curr):
        node = self.node
        goal = node.goal_distance()
        if node.goal_reached_check() and node.item_dropped_for_current_task==False:
            node.virtual_dropoff(node.task, node.current_item_id)
            node.item_dropped_for_current_task = True
            node.get_logger().info("Dropped item at Dock "+str(node.task))
        if not node.task_queue and node.item_dropped_for_current_task:
            if goal<node.docking_transition_distance:
                node.phase = 'docking' 
                node.docking_start_time = curr
                node.prepare_docking()
                return
        if node.item_dropped_for_current_task and node.task_queue:
            node.get_logger().info("The task has been completed, Remaining tasks: "+str(len(node.task_queue)))
            node.reset_task_state()
    

    def docking_state(self, curr):
        node = self.node
        if node.docking_complete:
            node.get_logger.info("Final docking complete after all tasks complete")
            node.delete_dock_box()
            node.reset_task_state()
            return

        if node.docking_start_time and curr-node.docking_start_time>node.max_docking_time:
            node.delete_dock_box()
            node.reset_task_state()