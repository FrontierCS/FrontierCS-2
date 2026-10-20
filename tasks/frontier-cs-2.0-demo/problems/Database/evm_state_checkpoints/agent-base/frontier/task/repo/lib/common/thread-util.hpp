#include<thread>

void PinRoundRobin(std::thread& thread, unsigned rotate_id);
void PinRoundRobin(std::jthread& thread, unsigned rotate_id);
