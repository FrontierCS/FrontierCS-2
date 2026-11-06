pragma solidity >=0.7.0 <0.9.0;

contract Smallbank {
    mapping(uint256=>uint256) savingStore;
    mapping(uint256=>uint256) checkingStore;

    function getBalance(uint256 addr) public returns (uint256 balance) {
        uint256 bal1 = savingStore[addr];
        uint256 bal2 = checkingStore[addr];
        
        balance = bal1 + bal2;
        return balance;
    }

    function depositChecking(uint256 addr, uint256 bal) public {
        uint256 bal1 = checkingStore[addr];
        uint256 bal2 = bal;
        
        
        checkingStore[addr] = bal1 + bal2;
    }

    function writeCheck(uint256 addr, uint256 bal) public {
        uint256 bal1 = checkingStore[addr];

        uint256 amount = bal;
      
        if (amount <= bal1) {
            checkingStore[addr] = bal1 - amount;
        } else {
            checkingStore[addr] = 0;
        }
    }
    
    function transactSaving(uint256 addr, uint256 bal) public {
        uint256 bal1 = savingStore[addr];
        uint256 bal2 = bal;
        
        savingStore[addr] = bal1 + bal2;
    }

    function sendPayment(uint256 addr0, uint256 addr1, uint256 bal) public {
        uint256 bal1 = checkingStore[addr0];
        uint256 bal2 = checkingStore[addr1];
        uint256 amount = bal;

        if (bal1 >= amount) {
            bal1 -= amount;
            bal2 += amount;
            
            checkingStore[addr0] = bal1;
            checkingStore[addr1] = bal2;
        }
    }

    function amalgamate(uint256 addr0, uint256 addr1) public {
       uint bal1 = savingStore[addr0];
       uint bal2 = checkingStore[addr1];
       
       checkingStore[addr1] = 0;
       savingStore[addr0] = bal1 + bal2;
    } 

    function init(uint256 key, uint256 value) public {
        savingStore[key] = value;
        checkingStore[key] = value;
    }
}