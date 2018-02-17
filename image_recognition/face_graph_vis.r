## face cluster visualizaation

.libPaths(.libPaths()[2])


library(networkD3)

# Create fake data
src <- c("A", "A", "A", "A",
        "B", "B", "C", "C", "D")
target <- c("B", "C", "D", "J",
            "E", "F", "G", "H", "I")
networkData <- data.frame(src, target)

# Plot
simpleNetwork(networkData)

python -m http.server 8000

library(networkD3)
library(visNetwork)
library(dplyr)
data(MisLinks)
data(MisNodes)

MisNodes_derp = MisNodes %>% 
    rename("label"=name) %>% 
    mutate(id = seq_len(nrow(MisNodes))-1)
MisLinks_derp = MisLinks %>% 
    rename("from"=source, "to"=target)
	
N = 2
MisNodes_derp_sel = MisNodes_derp[0:N,]	
MisLinks_derp_sel = MisLinks_derp[0:N,]
test = visNetwork(MisNodes_derp_sel,MisLinks_derp_sel) 

picture_file = 'C:\\Users\\C35612.LAUNCHER\\WinPython-64bit-3.5.3.1Qt5\\scripts\\image_team'
sel_face = dir()[0:N]
url_faces = paste0("http://127.0.0.1:8000/image_team/",sel_face)
test_node = visNodes(test,shape = "image",image = list(url_faces))


broken_image = "http://cdn0.iconfinder.com/data/icons/octicons/1024/mark-github-128.png"

url_1 = "https://1.bp.blogspot.com/-u_Xh1dSVAHk/VtthmWM8KLI/AAAAAAAASSw/ccJPxavqS8I/s1600/smiley-with-pencil.jpg"
url_2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/SNice.svg/1200px-SNice.svg.png"
derp_url = list(url_1,url_2)

visNodes(test,shape = "image",image = derp_url, brokenImage = broken_image)# Does not work
visNodes(test,shape = "image",image = derp_url[[1]], brokenImage = broken_image)#Does work


#test_node = visNodes(test,shape = "image",)


