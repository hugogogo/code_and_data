#### This is the source code for plotting Figure 4&5 (paths of selected variables for NHANES dataset)
set.seed(123)
rm(list = ls())
load("path.RData")
load("diabetes.RData")
name <- raw$name[-raw$one_hot_idx]
rm(raw)
# Install ggplot2
library("ggplot2")
library("ggrepel")

cost_order <- rank(costs, ties.method = "random")
# our method with regular costs
type_ez <- type[inc_our]
fac_type <- unique(type_ez)
type_ez[type_ez == fac_type[1]] <- "Demographics (Cost = 2)"
type_ez[type_ez == fac_type[[2]]] <- "Examination (Cost = 5)"
type_ez[type_ez == fac_type[[3]]] <- "Laboratory (Cost = 9)"
type_ez[type_ez == fac_type[[4]]] <- "Questionnaire (Cost = 4)"
# reorder the factor levels for plotting
type_ez <- factor(type_ez)
type_ez <- factor(type_ez, levels = levels(type_ez)[c(1, 4, 2, 3)])

mydf <- data.frame(Cost = unique(measure_cost[1, ]),
                   AUC = unique(measure_accu[1, ]),
                   Type = type_ez,
                   Id = cost_order[inc_our],
                   VarCost = costs[inc_our])
mydf$Name <- paste(mydf$Id, ":", name[inc_our], sep = "")

pdf("path.pdf", height = 9, width = 10)
p <- ggplot(mydf, aes(Cost, AUC)) + geom_blank() + ggtitle("Our proposal") +
  geom_line(linetype = "dashed")+
  theme_classic(base_size = 20) + 
  scale_y_continuous(limits = c(0.53, 0.77), breaks = seq(from = 0.53,
                                  to = 0.78, by = 0.05)) +
  geom_point(aes(color = Type), size = 7) +
  scale_color_manual(name = NULL, values=c("#D1E5F0","#92C5DE","#4393C3","#2166AC"))+
  guides(color=guide_legend(direction='vertical')) +
  #guides(color=FALSE) +
  geom_point(aes(fill = Name), alpha = 0, data = mydf) +
  #geom_text(aes(label = Id), data = mydf, size = 4) + 
  geom_label_repel(aes(label = Id), data = mydf, size = 7, min.segment.length = 0.001, direction = "y") + 
  guides(fill=FALSE) +
  #scale_fill_discrete(name = NULL, breaks = mydf$Name[order(mydf$Id)]) +
  theme(legend.key = element_blank(),
        legend.key.width = unit(0, "cm"),
        legend.title.align=0.5,
        legend.position = c(0.6, 0.4),
        legend.direction = "vertical",
        legend.box = "vertical")
p
dev.off()


# Katsevich&Ramdas(2018) with regular costs
type_ez <- type[inc_uw]
fac_type <- unique(type_ez)
type_ez[type_ez == "Demographics (real)"] <- "Demographics(2)"
type_ez[type_ez == "Examination (real)"] <- "Examination(5)"
type_ez[type_ez == "Laboratory (real)"] <- "Laboratory(9)"
type_ez[type_ez == "Questionnaire (real)"] <- "Questionnaire(4)"
# reorder the factor levels for plotting
type_ez <- factor(type_ez)
type_ez <- factor(type_ez, levels = levels(type_ez)[c(1, 4, 2, 3)])

mydf <- data.frame(Cost = unique(measure_cost[2, ]),
                   AUC = unique(measure_accu[2, ]),
                   Type = type_ez,
                   Id = cost_order[inc_uw],
                   VarCost = costs[inc_uw])
mydf$Name <- paste(mydf$Id, ":", name[inc_uw], sep = "")

pdf("path_uw.pdf", height = 9, width = 10)
p <- ggplot(mydf, aes(Cost, AUC)) + geom_blank() + ggtitle("Katsevich&Ramdas(2018)") +
  geom_line(linetype = "dashed")+
  theme_classic(base_size = 20) + 
  scale_y_continuous(limits = c(0.53, 0.77), breaks = seq(from = 0.53,
                                                          to = 0.78, by = 0.05)) +
  geom_point(aes(color = Type), size = 7) +
  scale_color_manual(name = NULL, values=c("#D1E5F0","#92C5DE","#4393C3","#2166AC"))+
  guides(color=FALSE) +
  geom_point(aes(fill = Name), alpha = 0, data = mydf) +
  #geom_text(aes(label = Id), data = mydf, size = 4) + 
  geom_label_repel(aes(label = Id), data = mydf, size = 7, min.segment.length = 0.001, direction = "y") + 
  scale_fill_discrete(name = NULL, breaks = mydf$Name[order(mydf$Id)]) +
  theme(legend.key = element_blank(),
        legend.key.width = unit(0, "cm"),
        legend.title.align=0.5,
        legend.position = c(0.6, 0.4),
        legend.direction = "vertical",
        legend.box = "vertical")
p
dev.off()

# Our method with regular costs
type_ez <- type[inc_our_exp]
fac_type <- unique(type_ez)
type_ez[type_ez == "Demographics (real)"] <- "Demographics (Cost = 4)"
type_ez[type_ez == "Examination (real)"] <- "Examination (Cost = 25)"
type_ez[type_ez == "Laboratory (real)"] <- "Laboratory (Cost = 81)"
type_ez[type_ez == "Questionnaire (real)"] <- "Questionnaire (Cost = 16)"
# reorder the factor levels for plotting
type_ez <- factor(type_ez)
type_ez <- factor(type_ez, levels = levels(type_ez)[c(1, 4, 2, 3)])

mydf <- data.frame(Cost = unique(measure_cost[3, ]),
                   AUC = unique(measure_accu[3, ]),
                   Type = type_ez,
                   Id = cost_order[inc_our_exp],
                   VarCost = costs[inc_our_exp])
mydf$Name <- paste(mydf$Id, ":", name[inc_our_exp], sep = "")

pdf("path_exp.pdf", height = 9, width = 10)
p <- ggplot(mydf, aes(Cost, AUC)) + geom_blank() + ggtitle("Our proposal (squared costs)") +
  geom_line(linetype = "dashed")+
  theme_classic(base_size = 20) + 
  scale_y_continuous(limits = c(0.53, 0.77), breaks = seq(from = 0.53,
                                                          to = 0.78, by = 0.05)) +
  geom_point(aes(color = Type), size = 7) +
  scale_color_manual(name = NULL, values=c("#D1E5F0","#92C5DE","#4393C3","#2166AC"))+
  guides(color=guide_legend(direction='vertical')) +
  geom_point(aes(fill = Name), alpha = 0, data = mydf) +
  #geom_text(aes(label = Id), data = mydf, size = 4) + 
  geom_label_repel(aes(label = Id), data = mydf, size = 7, min.segment.length = 0.001, direction = "y") + 
  scale_fill_discrete(name = NULL, breaks = mydf$Name[order(mydf$Id)]) +
  theme(legend.key = element_blank(),
        legend.key.width = unit(0, "cm"),
        legend.title.align=0.5,
        legend.position = c(0.6, 0.4),
        legend.direction = "vertical",
        legend.box = "horizontal")
p
dev.off()

